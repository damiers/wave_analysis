import numpy as np
import tifffile
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import TruncatedSVD
from sklearn.isotonic import IsotonicRegression
from typing import Union
import torch
import warnings
import math
import random
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

warnings.filterwarnings('ignore')

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


# Function for inpainting all frames in videos
def inpaint_video(video, inpaint_mask, n_components=20, inpaint_radius=10):
    """inpaint video

    Args:
        video (ndarray): T*w*h
        inpaint_mask (ndarray): w*h, masks for the brain area 
        n_components (int, optional): truncated svd. Defaults to 20.
        inpaint_radius (int, optional): controls the smoothness of the inpainted area. Defaults to 10.

    Returns:
        video_inpained (ndarray): T*w*h, inpained video
    """
    T, w, h = video.shape
    mask = np.logical_or(np.isnan(video[0]), inpaint_mask)  # inpainted area
    Y = video[:, ~mask]

    # run truncated SVD to get spatial footprints and inpaint each spatial footprints separately
    svd = TruncatedSVD(n_components)
    svd.fit(Y[::20])  # use fewer frames to achieve faster computation.
    V = svd.components_  # spatial footprints
    coefs = np.dot(Y, V.transpose(
    ))  # temporal components correspond to each spatial footprints

    # inpaint spatial components
    V_full = np.zeros((V.shape[0], w, h))
    V_full[:, ~mask] = V

    for i in range(V_full.shape[0]):
        vmin, vmax = V[i].min(), V[i].max()
        image = ((V_full[i] - vmin) / (vmax - vmin) * 65536).astype(np.uint16)
        image[mask] = 0
        V_full[i] = cv2.inpaint(
            image, inpaint_mask.astype(np.uint8), inpaint_radius,
            cv2.INPAINT_NS) / 65536.0 * (vmax - vmin) + vmin

    video_inpainted = video.copy()
    video_inpainted[:,
                    inpaint_mask > 0] = (coefs @ V_full[:, inpaint_mask > 0])
    return video_inpainted


# Function for velocity calculation based on phase
class calculate_velocity:

    def __init__(
        self,
        y,
        Fs=1,
        window='hann',
        nperseg=256,
        noverlap=128,
        freq_band=0.02,
        phase_nan=True,
    ) -> None:
        self.y = y
        self.Fs = Fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.freq_band = freq_band
        self.phase_nan = phase_nan
        # Normalize the cutoff frequency
        self.normalized_cutoff = self.freq_band / (0.5 * self.Fs)

    def spectrogram(self):
        freq, time, spectrogram = signal.spectrogram(self.y,
                                                     fs=self.Fs,
                                                     window=self.window,
                                                     nperseg=self.nperseg,
                                                     noverlap=self.noverlap)
        return freq, time, spectrogram

    def low_pass_filtering(self):
        # Design the low-pass filter
        b, a = signal.butter(4,
                             self.normalized_cutoff,
                             btype='low',
                             analog=False)
        filtered_signal = signal.filtfilt(b, a, self.y)
        return filtered_signal

    def hilbert_transform(self):
        # hilbert transform
        filtered_signal = self.low_pass_filtering()
        filtered_signal -= np.ptp(filtered_signal) / 2.0
        hilbert_signal = signal.hilbert(filtered_signal)
        return hilbert_signal

    def isotonic_phase(self, phase=None):
        if phase is None:
            phase = np.unwrap(np.angle(self.hilbert_transform()))
        # print(np.any(np.isnan(phase)), np.any(np.isinf(phase)))
        phase = np.nan_to_num(phase,
                              nan=0.0,
                              posinf=np.finfo(np.float32).max,
                              neginf=np.finfo(np.float32).min)
        iso_reg = IsotonicRegression().fit(np.arange(len(phase)), phase)
        phase_pred = iso_reg.predict((np.arange(len(phase))))
        return phase_pred

    # main function of this Class
    def preprocess(self):

        # low pass filtering
        b, a = signal.butter(4,
                             self.normalized_cutoff,
                             btype='low',
                             analog=False)
        if len(self.y.shape) == 1:
            self.y = self.y[:, np.newaxis]
        y_filtered = signal.filtfilt(b, a, self.y, axis=0)

        # hilbert trnasform
        y_filtered -= np.ptp(y_filtered, axis=0) / 2.0
        y_hilbert = signal.hilbert(y_filtered, axis=0)

        # isotonic phase
        phase = np.unwrap(np.angle(y_hilbert), axis=0)
        if len(phase.shape) == 2:
            for i in range(phase.shape[1]):
                phase[:, i] = self.isotonic_phase(phase[:, i])
                # nan in unnecessary timesteps
                # if self.phase_nan:
                #     if i < 1000 or i > 1500:
                #         phase[:, i] = np.nan
        else:
            for i in range(phase.shape[1]):
                for j in range(phase.shape[2]):
                    phase[:, i, j] = self.isotonic_phase(phase[:, i, j])

        return y_hilbert.squeeze(), phase.squeeze()


# Function of maximizing the display of phase prominence
class PhaseProcessor:
    """A class to process phase data
    """

    def __init__(self, threshold=0.1, if_vis=False):

        self.threshold = threshold
        self.if_vis = if_vis

    def process_pixel_phase(self, phase):

        angular_freq = np.diff(
            phase) * 20  # multiply 20 to convert to Hz. bin size is 50 ms

        longest_duration = 0
        longest_start_time = None
        longest_end_time = None

        start_time = None
        for i in range(len(angular_freq)):
            if angular_freq[i] >= self.threshold and start_time is None:
                start_time = i
            elif angular_freq[i] < self.threshold and start_time is not None:
                end_time = i
                duration = end_time - start_time
                if duration > longest_duration:
                    longest_duration = duration
                    longest_start_time = start_time
                    longest_end_time = end_time
                start_time = None

        if start_time is not None and len(angular_freq) > 0:
            end_time = len(angular_freq) - 1
            duration = end_time - start_time
            if duration > longest_duration:
                longest_duration = duration
                longest_start_time = start_time
                longest_end_time = end_time

        if self.if_vis:
            plt.figure(figsize=(10, 2))
            plt.plot(angular_freq, color='red')
            plt.axvline(x=longest_start_time)
            plt.axvline(x=longest_end_time)
            plt.show()
            print(
                f"Longest Time Period: {longest_start_time} to {longest_end_time}"
            )

        return longest_start_time, longest_end_time

    def pixel_phase(self, inpaint_data, pos):
        phase_process = np.zeros_like(inpaint_data)

        for i in range(pos.shape[1]):
            st, et = self.process_pixel_phase(inpaint_data[:, pos[:, i][0],
                                                           pos[:, i][1]])
            phase_process[st:et, pos[:, i][0],
                          pos[:, i][1]] = inpaint_data[st:et, pos[:, i][0],
                                                       pos[:, i][1]]

        return phase_process


def flow_to_image(flow: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert flow into middlebury color code image

    Parameters
    ----------
    flow: np.ndarray
        Optical flow map, height, width, channel
    
    Returns
    -------
    np.ndarray optical flow image in middlebury color
    """
    if torch.is_tensor(flow):
        flow = flow.cpu().detach().permute(1, 2, 0).numpy()

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    # rgb2gray
    #img = np.dot(np.asarray(img), [0.2989, 0.5870, 0.1140])

    return np.uint8(img)


def compute_color(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel() -> np.ndarray:
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG,
               0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB,
               1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR,
               2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    # rgb->gray
    #graywheel = np.dot(np.asarray(colorwheel), [0.2989, 0.5870, 0.1140])

    return colorwheel  #graywheel


def find_original(mod_phase,
                  sign,
                  fix,
                  st=1050,
                  et=1450,
                  hmin=0,
                  hmax=359,
                  wmin=0,
                  wmax=359,
                  recurrent_times=15,
                  top=10,
                  continue_t_len_down=100,
                  continue_t_len_up=1000):

    phase = mod_phase[:, hmin:hmax, wmin:wmax].copy()
    point_set = []
    continue_time = []
    origin_point_set = []
    Max_t = []

    points = list(
        np.asarray(
            np.nonzero(~np.isnan(fix[0, hmin:hmax, wmin:wmax]))).transpose(
                1, 0))

    if sign == 'iAdo2m':
        for p in points:
            if phase[:st, p[0], p[1]].max() < 0 and phase[
                    et, p[0], p[1]] > 0 and phase[:st, p[0],
                                                  p[1]].max() <= np.pi:
                max = phase[et:, p[0], p[1]].max()
                min = phase[:st, p[0], p[1]].min()
                max_t = np.where(phase[:, p[0], p[1]] == max)[0]
                min_t = np.where(phase[:, p[0], p[1]] == min)[0][-1]
                # print(max_t, min_t)
                if min_t >= 200 and max_t[-1] >= 2800:
                    point_set.append(np.asarray([p[0], p[1]]))
                    continue_time.append(max_t[-1] - max_t[0])
                    Max_t.append(max_t[0])

        above_threshold_indices = [
            idx for idx, value in enumerate(continue_time)
            if value > continue_t_len_down
        ]

        new_points = []
        new_max = []
        for index in above_threshold_indices:
            new_points.append(point_set[index])
            new_max.append(Max_t[index])
        top_k_max_t = sorted(new_max)[:top]
        for idx in range(len(new_max)):
            if new_max[idx] in top_k_max_t:
                origin_point_set.append(new_points[idx])

    elif sign == 'jRGEC01a':
        for p in points:
            if phase[:st, p[0], p[1]].max() < 0 and phase[
                    et, p[0], p[1]] > 0 and phase[:st, p[0],
                                                  p[1]].max() <= np.pi:
                max = phase[et:, p[0], p[1]].max()
                min = phase[:st, p[0], p[1]].min()
                max_t = np.where(phase[:, p[0], p[1]] == max)[0]
                min_t = np.where(phase[:, p[0], p[1]] >= 3)[0][0]
                # print(max_t, min_t)
                if min_t >= st and max_t[0] >= et and max_t[-1] <= 3000:
                    point_set.append(np.asarray([p[0], p[1]]))
                    continue_time.append(max_t[-1] - min_t)
                    Max_t.append(max_t[-1])

        above_threshold_indices = [
            idx for idx, value in enumerate(continue_time)
            if ((value < continue_t_len_up) and (value > continue_t_len_down))
        ]

        new_points = []
        new_max = []
        for index in above_threshold_indices:
            new_points.append(point_set[index])
            new_max.append(Max_t[index])
        top_k_max_t = sorted(new_max)[::-1][:top]
        top_k_min_t = sorted(new_max)[:top]
        for idx in range(len(new_max)):
            if new_max[idx] in top_k_max_t + top_k_min_t:
                origin_point_set.append(new_points[idx])

    else:
        raise ValueError('Please tell me which item should be analysed...')

    final_origin = []
    for ele in origin_point_set:
        final_origin.append(np.array([ele[0] + hmin, ele[1] + wmin]))

    return final_origin


def select_origin_region(origins, radius=10.0, method='hierarchy', wave_num=1):
    if method == 'ckdtree':
        kd_tree = cKDTree(np.asarray(origins))
        pairs_within_radius = kd_tree.query_pairs(radius)
        print(pairs_within_radius)

        non_isolated_points = set()
        for i, j in pairs_within_radius:
            non_isolated_points.add(tuple(origins[i]))
            non_isolated_points.add(tuple(origins[j]))
        non_isolated_points = np.array(list(non_isolated_points))

        return non_isolated_points

    elif method == 'hierarchy':
        Z = linkage(origins, method='complete')
        clusters = fcluster(Z, radius, criterion='distance')

        cluster_dict = {}
        for i in range(len(origins)):
            cluster_label = clusters[i]
            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = []
            cluster_dict[cluster_label].append(origins[i])
        # for key, value in cluster_dict.items():
        #     print(f"Cluster {key}: {value}")

        sorted_dict = dict(
            sorted(cluster_dict.items(), key=lambda x: len(x[1]),
                   reverse=True))

        if wave_num == 1:
            return np.asarray(list(sorted_dict.values())[0])
        else:
            return np.asarray(
                tuple(
                    np.asarray(list(sorted_dict.values())[i])
                    for i in range(wave_num)))

    else:
        raise ValueError('Please enter a valid cluster method...')


def min_circle(points):

    def is_inside(circle, point):
        distance = math.hypot(circle[0] - point[0], circle[1] - point[1])
        return distance <= circle[2]

    def update_circle(circle, point):
        center_x = (circle[0] + point[0]) / 2
        center_y = (circle[1] + point[1]) / 2
        radius = math.hypot(center_x - point[0], center_y - point[1])
        return (center_x, center_y, radius)

    random_point = random.choice(points)
    min_circle = (random_point[0], random_point[1], 0)

    for p in points:
        if not is_inside(min_circle, p):
            min_circle = update_circle(min_circle, p)

    return min_circle


def vis_wave(image,
             phase,
             idx,
             wave_x,
             wave_y,
             origin_region,
             wave_num,
             title_name=None,
             mag=None,
             thre=3,
             step=10,
             phase_step=10,
             linewidth=1.0,
             head_width=1.5,
             head_length=2,
             save=False,
             quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp, cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre and image[1050 + idx, i, j] > 0.4:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    if wave_num == 1:
        Points = origin_region.copy()
        Points[:, [0, 1]] = Points[:, [1, 0]]
        center_x, center_y, radius = min_circle(Points)
        center_x = int(center_x)
        center_y = int(center_y)

        ax.scatter(Points[:, 0], Points[:, 1], color='red', label='Origins')
        circle = plt.Circle((center_x, center_y),
                            5,
                            color='white',
                            fill=False,
                            linewidth=2,
                            label='Original Region')
        plt.annotate('Origin', (center_x, center_y),
                     textcoords='offset points',
                     xytext=(15, 10),
                     fontsize=16,
                     ha='center',
                     color='white')
        ax.add_patch(circle)

        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)
        # print(center_x, center_y,endpoints)
        for e in range(
                0,
                len(endpoints),
        ):
            u = (endpoints[e, 0] - center_x) / 4
            v = (endpoints[e, 1] - center_y) / 4
            ax.arrow(center_x,
                     center_y,
                     u,
                     v,
                     fc='green',
                     ec='green',
                     linewidth=linewidth,
                     head_width=head_width,
                     head_length=head_length,
                     length_includes_head=False)

    elif wave_num > 1:
        Center_x = np.zeros(wave_num)
        Center_y = np.zeros(wave_num)
        for i in range(wave_num):
            Points = origin_region[i].copy()
            Points[:, [0, 1]] = Points[:, [1, 0]]
            center_x, center_y, radius = min_circle(Points)
            Center_x[i] = int(center_x)
            Center_y[i] = int(center_y)

            ax.scatter(Points[:, 0],
                       Points[:, 1],
                       color='red',
                       label='Origins')

        center_x = int(np.mean(Center_x))
        center_y = int(np.mean(Center_y))
        circle = plt.Circle((center_x, center_y),
                            5,
                            color='white',
                            fill=False,
                            linewidth=2,
                            label='Original Region')
        plt.annotate('Origin', (center_x, center_y),
                     textcoords='offset points',
                     xytext=(15, 10),
                     fontsize=16,
                     ha='center',
                     color='white')
        ax.add_patch(circle)

        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)
        # print(center_x, center_y,endpoints)
        for e in range(
                0,
                len(endpoints),
        ):
            u = (endpoints[e, 0] - center_x) / 4
            v = (endpoints[e, 1] - center_y) / 4
            ax.arrow(center_x,
                     center_y,
                     u,
                     v,
                     fc='green',
                     ec='green',
                     linewidth=linewidth,
                     head_width=head_width,
                     head_length=head_length,
                     length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx}')

    plt.show()


def vis_wave_local(image,
                   phase,
                   idx,
                   wave_x,
                   wave_y,
                   origin_region,
                   wave_num,
                   sign=None,
                   title_name=None,
                   mag=None,
                   thre=3,
                   step=10,
                   phase_step=10,
                   linewidth=1.0,
                   head_width=1.5,
                   head_length=2,
                   save=False,
                   quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp[60:180, 200:320],
                   extent=(200, 320, 180, 60),
                   cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    if wave_num == 1:
        Points = origin_region.copy()
        Points[:, [0, 1]] = Points[:, [1, 0]]
        center_x, center_y, radius = min_circle(Points)
        center_x = int(center_x)
        center_y = int(center_y)

        # ax.scatter(Points[:, 0], Points[:, 1], color='green', label='Origins')
        circle = plt.Circle((center_x, center_y),
                            3,
                            color='white',
                            fill=False,
                            linewidth=2,
                            label='Original Region')
        plt.annotate('Origin', (center_x, center_y),
                     textcoords='offset points',
                     xytext=(15, 10),
                     fontsize=16,
                     ha='center',
                     color='white')
        ax.add_patch(circle)

        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)
        # print(center_x, center_y,endpoints)
        for i in range(0, len(endpoints)):
            u = (endpoints[i, 0] - center_x) / 4
            v = (endpoints[i, 1] - center_y) / 4
            ax.arrow(center_x,
                     center_y,
                     u,
                     v,
                     fc='green',
                     ec='green',
                     linewidth=linewidth,
                     head_width=head_width,
                     head_length=head_length,
                     length_includes_head=False)

    elif wave_num > 1:
        Center_x = np.zeros(wave_num)
        Center_y = np.zeros(wave_num)
        for i in range(wave_num):
            Points = origin_region[i].copy()
            Points[:, [0, 1]] = Points[:, [1, 0]]
            center_x, center_y, radius = min_circle(Points)
            Center_x[i] = int(center_x)
            Center_y[i] = int(center_y)

            # ax.scatter(Points[:, 0], Points[:, 1], color='green', label='Origins')

        center_x = int(np.mean(Center_x))
        center_y = int(np.mean(Center_y))
        circle = plt.Circle((center_x, center_y),
                            3,
                            color='white',
                            fill=False,
                            linewidth=2,
                            label='Original Region')
        plt.annotate('Origin', (center_x, center_y),
                     textcoords='offset points',
                     xytext=(15, 10),
                     fontsize=16,
                     ha='center',
                     color='white')
        ax.add_patch(circle)

        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)
        # print(center_x, center_y, endpoints)
        for i in range(0, len(endpoints)):
            u = (endpoints[i, 0] - center_x) / 4
            v = (endpoints[i, 1] - center_y) / 4
            ax.arrow(center_x,
                     center_y,
                     u,
                     v,
                     fc='green',
                     ec='green',
                     linewidth=linewidth,
                     head_width=head_width,
                     head_length=head_length,
                     length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx} (local)')

    plt.show()

    return center_x, center_y


def vis_wave_refine(image,
                    phase,
                    idx,
                    wave_x,
                    wave_y,
                    center_x,
                    center_y,
                    title_name=None,
                    mag=None,
                    thre=3,
                    step=10,
                    phase_step=10,
                    linewidth=1.0,
                    head_width=1.5,
                    head_length=2,
                    save=False,
                    quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp, cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre and image[1050 + idx, i, j] > 0.4:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    center_x = int(center_x)
    center_y = int(center_y)

    # ax.scatter(Points[:, 0], Points[:, 1], color='red', label='Origins')
    circle = plt.Circle((center_x, center_y),
                        5,
                        color='white',
                        fill=False,
                        linewidth=2,
                        label='Original Region')
    plt.annotate('Origin', (center_x, center_y),
                 textcoords='offset points',
                 xytext=(15, 10),
                 fontsize=16,
                 ha='center',
                 color='white')
    ax.add_patch(circle)

    endpoints = np.argwhere(
        abs(phase[idx + phase_step] -
            phase[idx, center_y, center_x]) < quiver_thre)
    # print(center_x, center_y,endpoints)
    for e in range(
            0,
            len(endpoints),
    ):
        u = (endpoints[e, 0] - center_x) / 4
        v = (endpoints[e, 1] - center_y) / 4
        ax.arrow(center_x,
                 center_y,
                 u,
                 v,
                 fc='green',
                 ec='green',
                 linewidth=linewidth,
                 head_width=head_width,
                 head_length=head_length,
                 length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx}')

    plt.show()


def vis_wave_local_refine(image,
                          phase,
                          idx,
                          wave_x,
                          wave_y,
                          center_x,
                          center_y,
                          title_name=None,
                          mag=None,
                          thre=3,
                          step=10,
                          phase_step=10,
                          linewidth=1.0,
                          head_width=1.5,
                          head_length=2,
                          save=False,
                          quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp[60:180, 200:320],
                   extent=(200, 320, 180, 60),
                   cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    center_x = int(center_x)
    center_y = int(center_y)
    # ax.scatter(Points[:, 0], Points[:, 1], color='green', label='Origins')
    circle = plt.Circle((center_x, center_y),
                        3,
                        color='white',
                        fill=False,
                        linewidth=2,
                        label='Original Region')
    plt.annotate('Origin', (center_x, center_y),
                 textcoords='offset points',
                 xytext=(15, 10),
                 fontsize=16,
                 ha='center',
                 color='white')
    ax.add_patch(circle)

    endpoints = np.argwhere(
        abs(phase[idx + phase_step] -
            phase[idx, center_y, center_x]) < quiver_thre)
    # print(center_x, center_y,endpoints)
    for i in range(0, len(endpoints)):
        u = (endpoints[i, 0] - center_x) / 4
        v = (endpoints[i, 1] - center_y) / 4
        ax.arrow(center_x,
                 center_y,
                 u,
                 v,
                 fc='green',
                 ec='green',
                 linewidth=linewidth,
                 head_width=head_width,
                 head_length=head_length,
                 length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx} (local)')
        plt.savefig(f'../{title_name}_origin_flow_local_align_{1050+idx}.pdf')

    plt.show()


def vis_wave_local_standard(image,
                            phase,
                            idx,
                            wave_x,
                            wave_y,
                            center_x,
                            center_y,
                            title_name=None,
                            mag=None,
                            thre=3,
                            step=10,
                            phase_step=10,
                            linewidth=1.0,
                            head_width=1.5,
                            head_length=2,
                            save=False,
                            quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp[60:180, 200:320],
                   extent=(200, 320, 180, 60),
                   cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    center_x = int(center_x)
    center_y = int(center_y)
    # ax.scatter(Points[:, 0], Points[:, 1], color='green', label='Origins')
    circle = plt.Circle((center_x, center_y),
                        3,
                        color='white',
                        fill=False,
                        linewidth=2,
                        label='Original Region')
    plt.annotate('Origin', (center_x, center_y),
                 textcoords='offset points',
                 xytext=(15, 10),
                 fontsize=16,
                 ha='center',
                 color='white')
    ax.add_patch(circle)

    endpoints = np.argwhere(
        abs(phase[idx + phase_step] -
            phase[idx, center_y, center_x]) < quiver_thre)
    # print(center_x, center_y,endpoints)
    # for i in range(0, len(endpoints)):
    # u = (endpoints[i,0] - center_x) / 4
    # v = (endpoints[i,1] - center_y) / 4
    # print(u, v)
    # ax.arrow(center_x, center_y, u, v, fc='green', ec='green', linewidth=linewidth, head_width=head_width, head_length=head_length, length_includes_head=False)
    DX = [-5, -45, -60, -45, -5, 10]
    DY = [-10, -25, 10, 60, 65, 40]
    for i in range(len(DX)):
        ax.arrow(center_x,
                 center_y,
                 dx=DX[i],
                 dy=DY[i],
                 fc='green',
                 ec='green',
                 linewidth=linewidth,
                 head_width=head_width,
                 head_length=head_length,
                 length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx} (local)')
        plt.axis('off')

    plt.show()


def find_min_angle_vec(standard_vec, vec_list):

    angle = []
    for i in range(len(vec_list)):
        # print(standard_vec, vec_list[i])

        l = np.sqrt(np.asarray(standard_vec).dot(
            np.asarray(standard_vec))) * np.sqrt(
                np.asarray(vec_list[i]).dot(np.asarray(vec_list[i])))
        cos = np.asarray(standard_vec).dot(np.asarray(vec_list[i])) / l

        angle.append(np.arccos(cos))

    return np.sqrt(
        np.asarray(vec_list[np.argmin(np.asarray(angle))]).dot(
            np.asarray(vec_list[np.argmin(np.asarray(angle))])))


def vis_wave_local_direct(image,
                          phase,
                          idx,
                          wave_x,
                          wave_y,
                          center_x,
                          center_y,
                          title_name=None,
                          mag=None,
                          thre=3,
                          step=10,
                          phase_step=10,
                          linewidth=1.0,
                          head_width=1.5,
                          head_length=2,
                          save=False,
                          quiver_thre=0.01):
    t, h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots(figsize=(8, 6))
    image_temp = image[1050 + idx].copy()
    image_temp[np.isnan(image_temp)] = 0
    im = ax.imshow(image_temp[60:180, 200:320],
                   extent=(200, 320, 180, 60),
                   cmap='jet')
    cbar = plt.colorbar(im,
                        ax=ax,
                        orientation='horizontal',
                        pad=0.1,
                        shrink=0.6)
    cbar.set_label('delta F/F0', rotation=0, labelpad=10)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='white',
                         ec='white',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    center_x = int(center_x)
    center_y = int(center_y)
    # ax.scatter(Points[:, 0], Points[:, 1], color='green', label='Origins')
    circle = plt.Circle((center_x, center_y),
                        3,
                        color='white',
                        fill=False,
                        linewidth=2,
                        label='Original Region')
    plt.annotate('Origin', (center_x, center_y),
                 textcoords='offset points',
                 xytext=(15, 10),
                 fontsize=16,
                 ha='center',
                 color='white')
    ax.add_patch(circle)

    endpoints = np.argwhere(
        abs(phase[idx + phase_step] -
            phase[idx, center_y, center_x]) < quiver_thre)
    final_points = []
    # print(center_x, center_y,endpoints)
    for i in range(0, len(endpoints)):
        u = (endpoints[i, 0] - center_x) / 4
        v = (endpoints[i, 1] - center_y) / 4
        final_points.append([u, v])
        # print(u, v)
        # ax.arrow(center_x, center_y, u, v, fc='green', ec='green', linewidth=linewidth, head_width=head_width, head_length=head_length, length_includes_head=False)
    # DX = [0]#[-5, -45, -60, -45, -5, 10]
    # DY = [1]#[-10, -25, 10, 60, 65, 40]
    standard = [[0, 1], [0.2, 0.5], [-1, 0], [-0.8, -1], [-0.5, 0.5],
                [-0.7, 0.3], [-0.3, 0.7]]

    for i in range(len(standard)):
        l = find_min_angle_vec(standard[i], final_points)
        # print(l)

        dx = standard[i][0] * l / np.sqrt(
            np.asarray(standard[i]).dot(np.asarray(standard[i])))
        dy = standard[i][1] * l / np.sqrt(
            np.asarray(standard[i]).dot(np.asarray(standard[i])))
        # print(l, dx, dy)

        ax.arrow(center_x,
                 center_y,
                 dx,
                 dy,
                 fc='green',
                 ec='green',
                 linewidth=linewidth,
                 head_width=head_width,
                 head_length=head_length,
                 length_includes_head=False)

    if save:
        plt.title(f'{title_name} flow at origin t{1050+idx} (local)')
        # plt.axis('off')
        plt.savefig(
            f'../{title_name}_origin_flow_local_align_standard_direct_{1050+idx}.pdf'
        )

    plt.show()

    import tifffile


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Optional
import cv2
from matplotlib.animation import ArtistAnimation, FFMpegWriter

import warnings

warnings.filterwarnings('ignore')


# function of calculating wave speed
def wave_speed(data, idx, max_dt=5):

    data[data == 0] = np.nan

    T, H, W = data.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    dx, dy = np.meshgrid(np.arange(-10, 11), np.arange(-10, 11))
    x_neigh, y_neigh = x.reshape(-1, 1) + dx.reshape(1, -1), y.reshape(
        -1, 1) + dy.reshape(1, -1)

    idx_oob = (x_neigh < 0) | (x_neigh >= W) | (y_neigh < 0) | (
        y_neigh >= W)  # out of boundary pixels
    x_neigh[idx_oob] = 0
    y_neigh[idx_oob] = 0
    idx_neigh = y_neigh * H + x_neigh  # neighboring pixels of each pixel
    delta = (
        2 * np.pi / 200
    )  # if the phase difference is no bigger than delta, then they are equal
    data[data == 0] = np.nan
    data = np.mod(data + np.pi, 2 * np.pi) - np.pi

    vx, vy = np.zeros_like(data, dtype=np.float32), np.zeros_like(
        data, dtype=np.float32)  # velocity in x & t direction
    idx_select = idx_neigh[idx.flatten(), :]  # indices of neighboring pixels

    for t in range(max_dt, T - max_dt):
        frame_t = data[t][idx]  # the phase of selected pixels at frame t
        dx_sum, dy_sum, n = np.zeros_like(frame_t), np.zeros_like(
            frame_t), np.zeros_like(frame_t)
        for dt in range(-max_dt, max_dt + 1):
            if dt == 0:
                continue
            frame_tdt = data[t + dt].flatten()[
                idx_select]  # phase of neighboring pixels in frame t+dt
            flag_equal = (
                np.abs(frame_tdt - frame_t.reshape(-1, 1)) < delta
            )  # find neighboring pixels with the same value in frame (t+dt)
            dx_sum += flag_equal @ dx.flatten() / dt
            dy_sum += flag_equal @ dy.flatten() / dt
            n += flag_equal.sum(axis=1)
        n[n == 0] = np.inf
        vx[t][idx], vy[t][idx] = dx_sum * 1.0 / n, dy_sum * 1.0 / n
        # if np.mod(t, 10) == 0:
        #     print(t)

    mag = np.sqrt(vx.copy()**2 + vy.copy()**2)

    return vx, vy, mag


def vis_wave(image,
             idx,
             wave_x,
             wave_y,
             mag=None,
             thre=None,
             step=10,
             linewidth=0.5,
             head_width=1.5,
             head_length=2,
             save=False):
    h, w = image.shape
    if mag is None:
        mag = np.sqrt(wave_x.copy()**2 + wave_y.copy()**2)
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    fig, ax = plt.subplots()
    ax.imshow(image)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if mag[idx, i, j] > thre:
                ax.arrow(j,
                         i,
                         wave_x[idx, i, j],
                         wave_y[idx, i, j],
                         fc='r',
                         ec='r',
                         linewidth=linewidth,
                         head_width=head_width,
                         head_length=head_length,
                         length_includes_head=False)

    if save:
        plt.title(f'flow at t{idx}')
        plt.savefig(f'../flow_{idx}.pdf')


class FileProcessor:

    def __init__(self, base_path: str, file_names: List[str]):
        self.base_path = base_path
        self.file_names = file_names

    def check_files_exist(self) -> bool:
        for file_name in self.file_names:
            file_path = os.path.join(self.base_path, file_name)
            if not os.path.exists(file_path):
                return False
        return True

    def load_files(self) -> List[np.ndarray]:
        loaded_data = []
        for file_name in self.file_names:
            file_path = os.path.join(self.base_path, file_name)
            data = np.load(file_path)
            loaded_data.append(data)
        return loaded_data

    def save_files(self, data_list: List[np.ndarray]):
        for file_name, data in zip(self.file_names, data_list):
            file_path = os.path.join(self.base_path, file_name)
            np.save(file_path, data)
            print(f"Files are saved: {file_path}")

    def process(self, algorithm) -> Optional[List[np.ndarray]]:
        if self.check_files_exist():
            print("Files are existed. Load directly")
            return self.load_files()
        else:
            print("Files are not existed. Calculate them using algorithm.")
            computed_data = algorithm
            self.save_files(computed_data)
            return None


# function to make a flow video
def generate_video(image_list, output_file, fps=30):

    width, height = image_list[0].shape[1], image_list[0].shape[0]

    video_writer = cv2.VideoWriter(output_file,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height),
                                   isColor=True)

    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        video_writer.write(image)

    video_writer.release()
    print('Video is finished.')


# function to make a quiver video
def quiver_video(ori_data, vx, vy, filename, points, mag=None, step=10):
    arr_without_nan = ori_data[~np.isnan(ori_data)]
    min_value = np.nanmin(arr_without_nan)
    max_value = np.nanmax(arr_without_nan)

    h, w = vx.shape[1:]
    # if mag is None:
    #     mag = np.sqrt(vx.copy()**2 + vy.copy()**2)

    st = 1050
    et = 1450

    plt.ioff()

    fig, ax = plt.subplots(figsize=(w / 50, h / 50))

    writer = FFMpegWriter(fps=10)
    with writer.saving(fig, filename, dpi=300):
        for t in range(st, et):
            ax.clear()
            ax.imshow(ori_data[t], vmin=min_value, vmax=max_value)
            artists = []
            for p in points:
                i, j = p
                arrow = ax.arrow(j,
                                 i,
                                 vx[t - st, i, j],
                                 vy[t - st, i, j],
                                 fc='red',
                                 ec='red',
                                 linewidth=0.3,
                                 head_width=0.8,
                                 head_length=1.5,
                                 length_includes_head=False)
                artists.append(arrow)
                ax.add_artist(arrow)
                ax.axis('off')
                ax.set_aspect('equal')
            plt.tight_layout()
            writer.grab_frame()

    plt.close(fig)
    plt.ion()


# function to make polar histgram
def polar_hist(wave_x,
               wave_y,
               idx=None,
               color=None,
               thres=0.5,
               title_name=None,
               save=False):

    wave_x_flat = wave_x.flatten()
    wave_y_flat = wave_y.flatten()

    r = np.sqrt(wave_x_flat**2 + wave_y_flat**2)
    theta = np.arctan2(wave_y_flat, wave_x_flat)

    filtered_r = r[r > thres]
    filtered_theta = theta[r > thres]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    n, bins, _ = ax.hist(filtered_theta,
                         bins=100,
                         color=color,
                         edgecolor='black',
                         density=True)

    plt.gca().yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    plt.title(f'{title_name} wave direction at t{idx}')
    if save:
        plt.savefig(f'../{title_name}_wave_direction_at_t{idx}.pdf')
    plt.show()


# function to make polar histgram videos
def pre_hist(wave_x, wave_y, thres=2):

    total_filtered_theta = []
    current_frame_data_percentage = []
    for i in range(len(wave_x)):
        wave_x_flat = wave_x[i].flatten()
        wave_y_flat = wave_y[i].flatten()
        r = np.sqrt(wave_x_flat**2 + wave_y_flat**2)
        theta = np.arctan2(wave_y_flat, wave_x_flat)
        filtered_theta = theta[r > thres]
        total_filtered_theta.append(filtered_theta)

    total_len = 0
    count_list = []
    for j in range(len(total_filtered_theta)):
        count_list.append(len(total_filtered_theta[j]))
        total_len += len(total_filtered_theta[j])

    current_frame_data_percentage = [c / total_len for c in count_list]

    return total_filtered_theta, current_frame_data_percentage, total_len


def polar_hist_mp4(total_filtered_theta1,
                   total_filtered_theta2,
                   current_frame_data_percentage1,
                   current_frame_data_percentage2,
                   idx,
                   frame_number,
                   color1=None,
                   color2=None,
                   thres=2):

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_ylim(0, 1)
    n1, bins1, _ = ax1.hist(total_filtered_theta1[idx],
                            bins=20,
                            color=color1,
                            edgecolor='black',
                            density=False,
                            weights=np.full(
                                len(total_filtered_theta1[idx]),
                                current_frame_data_percentage1[idx]))
    # ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_ylim(0, 1)
    n2, bins2, _ = ax2.hist(total_filtered_theta2[idx],
                            bins=20,
                            color=color2,
                            edgecolor='black',
                            density=False,
                            weights=np.full(
                                len(total_filtered_theta2[idx]),
                                current_frame_data_percentage2[idx]))
    # ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))

    # plt.text(-np.pi/2, 1.2, f"{frame_number} frame", ha='center', fontsize=10)
    plt.suptitle(f'polar hist MP4 - {frame_number} frame', fontsize=16)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

    plt.close(fig)  # Close the figure to avoid memory leaks
    return image


## visualization functions


def compare_raw_inpaint(img_raw, img_inpaint, min_max=[0, 1.25]):
    """display the original image and inpainted image side by side
    in a single figure
    The original image is displayed on the left and the inpainted image
    is displayed on the right.

    Args:
        img_raw (2d array): raw image 
        img_inpaint (2d array): inpainted image 
        min_max (list, optional): [vmin, vmax]. Defaults to [0, 1.25].
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(img_raw, vmin=min_max[0], vmax=min_max[1])
    plt.colorbar()
    plt.axis('off')
    plt.title('original image')
    plt.subplot(122)
    plt.imshow(img_inpaint, vmin=min_max[0], vmax=min_max[1])
    plt.colorbar()
    plt.axis('off')
    plt.title('inpainted image')
    plt.show()


## SAVE results
def save_as_tiff(data, file_path):
    """save the data as a tiff file

    Args:
        data (ndarray): image data 
        file_path (str): file name

    Returns:
        boolean: succeed or not 
    """
    if os.path.exists(file_path):
        print(f"The file is already existing: {file_path}")
        return False
    try:
        tifffile.imwrite(file_path, data)
        print(f"Saved file: {file_path}")
        return True
    except Exception as e:
        print(f"Some errors happen: {e}")
        return False
