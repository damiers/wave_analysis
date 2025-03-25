import numpy as np
import tifffile
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
import os
from typing import List, Optional
from matplotlib.animation import ArtistAnimation, FFMpegWriter
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

import warnings

warnings.filterwarnings('ignore')

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


# Function for inpainting all frames in videos
def inpaint_video(video,
                  inpaint_mask,
                  n_components=20,
                  inpaint_radius=10,
                  inpaint_file=None):
    """inpaint video

    Args:
        video (ndarray): T*w*h
        inpaint_mask (ndarray): w*h, masks for the brain area 
        n_components (int, optional): truncated svd. Defaults to 20.
        inpaint_radius (int, optional): controls the smoothness of the inpainted area. Defaults to 10.
        inpaint_file (str, optional): file name to save the inpainted video. Defaults to None.

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


class calculate_phase:
    """
    Class for calculating phase using signal processing techniques.
        This class performs:
        1. Signal filtering
        2. Hilbert transform for phase extraction
        3. Phase unwrapping and isotonic regression
    
    Attributes:
        y (ndarray): Input signal
        Fs (float): Sampling frequency
        window (str): Window type for spectrogram
        nperseg (int): Length of each segment for spectrogram
        noverlap (int): Number of points to overlap between segments
        freq_band (float): Frequency band for filtering
        phase_nan (bool): Whether to set phase to nan in unnecessary timesteps
        normalized_cutoff (float): Normalized cutoff frequency for filter
    """

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
        """Initialize calculator with signal processing parameters.

        Args:
            y (ndarray): Input signal, shape (n_samples,) or (n_samples, n_channels)
            Fs (float, optional): Sampling frequency in Hz. Defaults to 1.
            window (str, optional): Window type for spectrogram. Defaults to 'hann'.
            nperseg (int, optional): Length of each segment for spectrogram. Defaults to 256.
            noverlap (int, optional): Number of points to overlap between segments. Defaults to 128.
            freq_band (float, optional): Frequency band for filtering in Hz. Defaults to 0.02.
            phase_nan (bool, optional): Whether to set phase to nan in unnecessary timesteps. Defaults to True.
        """
        self.y = y
        self.Fs = Fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.freq_band = freq_band
        self.phase_nan = phase_nan
        # Normalize the cutoff frequency to Nyquist frequency (0.5*Fs)
        self.normalized_cutoff = self.freq_band / (0.5 * self.Fs)

    def spectrogram(self):
        """Compute spectrogram of the input signal.

        Returns:
            tuple: Contains three elements:
                - freq (ndarray): Array of frequencies, shape (n_freqs,)
                - time (ndarray): Array of time points, shape (n_times,)
                - spectrogram (ndarray): Spectrogram of signal, shape (n_freqs, n_times)
        """
        freq, time, spectrogram = signal.spectrogram(self.y,
                                                     fs=self.Fs,
                                                     window=self.window,
                                                     nperseg=self.nperseg,
                                                     noverlap=self.noverlap)
        return freq, time, spectrogram

    def low_pass_filtering(self):
        """Apply low-pass Butterworth filter to the signal.

        Returns:
            ndarray: Filtered signal, same shape as input signal
        """
        # Design 4th order Butterworth low-pass filter
        b, a = signal.butter(4,
                             self.normalized_cutoff,
                             btype='low',
                             analog=False)
        # Apply zero-phase filtering
        filtered_signal = signal.filtfilt(b, a, self.y)
        return filtered_signal

    def hilbert_transform(self):
        """Compute analytic signal using Hilbert transform.

        Returns:
            ndarray: Complex analytic signal, same shape as input
        """
        # First apply low-pass filtering
        filtered_signal = self.low_pass_filtering()
        # Center signal around zero
        filtered_signal -= np.ptp(filtered_signal) / 2.0
        # Compute analytic signal
        hilbert_signal = signal.hilbert(filtered_signal)
        return hilbert_signal

    def isotonic_phase(self, phase=None):
        """Apply isotonic regression to phase signal to ensure monotonicity.

        Args:
            phase (ndarray, optional): Input phase signal. If None, uses Hilbert transform phase. 
                                      Defaults to None.

        Returns:
            ndarray: Monotonically increasing phase signal, same shape as input
        """
        if phase is None:
            # Get phase from Hilbert transform and unwrap discontinuities
            phase = np.unwrap(np.angle(self.hilbert_transform()))

        # Handle NaN and infinite values
        phase = np.nan_to_num(phase,
                              nan=0.0,
                              posinf=np.finfo(np.float32).max,
                              neginf=np.finfo(np.float32).min)

        # Fit isotonic regression to enforce monotonic phase
        iso_reg = IsotonicRegression().fit(np.arange(len(phase)), phase)
        phase_pred = iso_reg.predict((np.arange(len(phase))))
        return phase_pred

    def preprocess(self):
        """Main preprocessing pipeline for calculation.

        Performs:
        1. Low-pass filtering
        2. Hilbert transform
        3. Phase unwrapping
        4. Isotonic phase correction

        Returns:
            tuple: Contains two elements:
                - y_hilbert (ndarray): Analytic signal after Hilbert transform
                - phase (ndarray): Processed phase information
        """
        # Design low-pass filter
        b, a = signal.butter(4,
                             self.normalized_cutoff,
                             btype='low',
                             analog=False)

        # Ensure input is 2D (samples, channels)
        if len(self.y.shape) == 1:
            self.y = self.y[:, np.newaxis]

        # Apply zero-phase filtering
        y_filtered = signal.filtfilt(b, a, self.y, axis=0)

        # Center signal and compute analytic signal
        y_filtered -= np.ptp(y_filtered, axis=0) / 2.0
        y_hilbert = signal.hilbert(y_filtered, axis=0)

        # Unwrap phase and apply isotonic regression
        phase = np.unwrap(np.angle(y_hilbert), axis=0)

        # Handle 2D and 3D cases differently
        if len(phase.shape) == 2:
            for i in range(phase.shape[1]):
                phase[:, i] = self.isotonic_phase(phase[:, i])
        else:
            for i in range(phase.shape[1]):
                for j in range(phase.shape[2]):
                    phase[:, i, j] = self.isotonic_phase(phase[:, i, j])

        return y_hilbert.squeeze(), phase.squeeze()


class PhaseProcessor:
    """
    Class for processing and enhancing phase data to maximize display prominence.

    This class identifies and extracts the most prominent phase segments in signals based on:
        1. Angular frequency thresholding
        2. Duration analysis of valid segments
        3. Visualization capabilities for verification
    """

    def __init__(self, threshold=0.1, if_vis=False):
        """Initialize PhaseProcessor with processing parameters.

        Args:
            threshold (float, optional): Angular frequency threshold (in Hz) for 
                                        considering a segment as valid. Defaults to 0.1.
            if_vis (bool, optional): Whether to visualize processing results. Defaults to False.
        """
        self.threshold = threshold
        self.if_vis = if_vis

    def process_pixel_phase(self, phase):
        """Process phase data for a single pixel/time-series to find longest valid segment.

        Args:
            phase (ndarray): 1D array of phase values for a single pixel/time-series,
                              shape (n_timepoints,)

        Returns:
            tuple: Contains two elements:
                - longest_start_time (int): Start index of longest valid segment
                - longest_end_time (int): End index of longest valid segment
        """
        # Calculate angular frequency (derivative of phase)
        # Multiply by 20 to convert to Hz (assuming 50ms bins: 1/0.05 = 20)
        angular_freq = np.diff(phase) * 20

        # Initialize tracking variables
        longest_duration = 0
        longest_start_time = None
        longest_end_time = None
        start_time = None

        # Scan through angular frequency to find valid segments
        for i in range(len(angular_freq)):
            # Segment starts when frequency crosses threshold
            if angular_freq[i] >= self.threshold and start_time is None:
                start_time = i
            # Segment ends when frequency drops below threshold
            elif angular_freq[i] < self.threshold and start_time is not None:
                end_time = i
                duration = end_time - start_time
                # Update longest segment if current is longer
                if duration > longest_duration:
                    longest_duration = duration
                    longest_start_time = start_time
                    longest_end_time = end_time
                start_time = None

        # Handle case where segment continues until end of data
        if start_time is not None and len(angular_freq) > 0:
            end_time = len(angular_freq) - 1
            duration = end_time - start_time
            if duration > longest_duration:
                longest_duration = duration
                longest_start_time = start_time
                longest_end_time = end_time

        # Visualization if enabled
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
        """Process phase data for multiple pixels/positions.

        Args:
            inpaint_data (ndarray): 3D array of phase data, 
                                    shape (n_timepoints, height, width)
            pos (ndarray): 2D array of pixel positions to process, 
                           shape (n_pixels, 2) where each row is (y,x) coordinates

        Returns:
            ndarray: Processed phase data with only significant segments retained,
                    same shape as inpaint_data
        """
        # Initialize output array
        phase_process = np.zeros_like(inpaint_data)

        # Process each specified pixel position
        for i in range(pos.shape[1]):
            # Get start and end times for valid segment
            st, et = self.process_pixel_phase(inpaint_data[:, pos[:, i][0],
                                                           pos[:, i][1]])

            # Copy only the valid segment to output
            phase_process[st:et, pos[:, i][0], pos[:, i][1]] = \
                inpaint_data[st:et, pos[:, i][0], pos[:, i][1]]

        return phase_process


def flow_to_image(flow: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert flow into middlebury color code image

    Args:
    flow: [np.ndarray], Optical flow map, height, width, channel
    
    Returns: [np.ndarray], optical flow image in middlebury color
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
    """
    Convert 2D flow field (u,v) to RGB color image representation.
    
    This function visualizes optical flow or vector fields by encoding:
    - Direction as hue (based on angle)
    - Magnitude as saturation (with normalization)
    - Handles NaN values by masking them
    
    Color mapping follows the standard color wheel convention used in optical flow visualization.

    Args:
        u (np.ndarray): 2D array of horizontal components, shape (height, width)
        v (np.ndarray): 2D array of vertical components, shape (height, width)

    Returns:
        np.ndarray: RGB color image of the flow field, shape (height, width, 3), dtype uint8
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])  # Initialize output RGB image

    # Handle NaN values by setting to 0 and creating mask
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    # Get standard optical flow color wheel (predefined colors for different angles)
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)  # Number of colors in the wheel

    # Compute flow magnitude (vector length)
    rad = np.sqrt(u**2 + v**2)

    # Compute flow angle (direction) normalized to [-1,1] range
    a = np.arctan2(-v, -u) / np.pi

    # Map angle to colorwheel indices (1-based indexing)
    fk = (a + 1) / 2 * (ncols - 1) + 1

    # Get base color indices
    k0 = np.floor(fk).astype(int)

    # Get next color indices with circular wrapping
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1  # Wrap around color wheel
    f = fk - k0  # Fractional part for interpolation

    # Process each RGB channel
    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]

        # Get colors to interpolate between
        col0 = tmp[k0 - 1] / 255  # Convert from 0-255 to 0-1
        col1 = tmp[k1 - 1] / 255

        # Linear interpolation between colors
        col = (1 - f) * col0 + f * col1

        # Adjust saturation based on magnitude
        idx = rad <= 1  # For normalized magnitudes
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # Scale down saturation
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75  # Dim colors for large magnitudes

        # Apply NaN mask and convert to 8-bit
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def wave_speed(data, idx, max_dt=5):
    """Calculate wave propagation speed from phase data using spatiotemporal analysis.

    This function estimates wave velocity by analyzing phase propagation patterns across
    neighboring pixels over time. The method:
    1. Identifies phase fronts in the data
    2. Tracks their movement between frames
    3. Computes velocity vectors (vx, vy) and magnitude

    Args:
        data (np.ndarray): 3D array of phase values, shape (time, height, width)
        idx (np.ndarray): 1D array of pixel indices to analyze (flattened spatial indices)
        max_dt (int, optional): Maximum time delta to consider for velocity calculation. 
                               Defaults to 5.

    Returns:
        tuple: Three elements containing:
            - vx (np.ndarray): x-component of velocity, same shape as input data
            - vy (np.ndarray): y-component of velocity, same shape as input data
            - mag (np.ndarray): Velocity magnitude, same shape as input data
    """
    # Clean data by replacing zeros with NaN and normalizing phase to [-π, π]
    data[data == 0] = np.nan
    T, H, W = data.shape
    data = np.mod(data + np.pi, 2 * np.pi) - np.pi  # Wrap phase to [-π, π]

    # Create coordinate grids and neighborhood offsets
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    dx, dy = np.meshgrid(np.arange(-10, 11),
                         np.arange(-10, 11))  # 21x21 neighborhood

    # Calculate all possible neighbor positions
    x_neigh = x.reshape(-1, 1) + dx.reshape(1, -1)
    y_neigh = y.reshape(-1, 1) + dy.reshape(1, -1)

    # Handle out-of-bound neighbors by setting to (0,0)
    idx_oob = (x_neigh < 0) | (x_neigh >= W) | (y_neigh < 0) | (y_neigh >= H)
    x_neigh[idx_oob] = 0
    y_neigh[idx_oob] = 0

    # Convert neighbor coordinates to flattened indices
    idx_neigh = y_neigh * W + x_neigh  # Note: Fixed H to W for correct flattening

    # Phase difference threshold for considering phases equal
    delta = 2 * np.pi / 200  # ~1.8 degree threshold

    # Initialize velocity arrays
    vx = np.zeros_like(data, dtype=np.float32)
    vy = np.zeros_like(data, dtype=np.float32)
    idx_select = idx_neigh[
        idx.flatten(), :]  # Neighbor indices for selected pixels

    # Main processing loop over time frames
    for t in range(max_dt, T - max_dt):
        frame_t = data[t][idx]  # Current frame phase values at selected pixels

        # Initialize accumulators
        dx_sum = np.zeros_like(frame_t)
        dy_sum = np.zeros_like(frame_t)
        n = np.zeros_like(frame_t)  # Count of valid neighbors

        # Analyze temporal neighbors within ±max_dt
        for dt in range(-max_dt, max_dt + 1):
            if dt == 0:
                continue  # Skip current frame

            # Get phase values of neighbors in future/past frame
            frame_tdt = data[t + dt].flatten()[idx_select]

            # Find neighbors with similar phase (within delta)
            flag_equal = np.abs(frame_tdt - frame_t.reshape(-1, 1)) < delta

            # Accumulate displacements weighted by 1/dt (inverse time)
            dx_sum += flag_equal @ dx.flatten() / dt
            dy_sum += flag_equal @ dy.flatten() / dt
            n += flag_equal.sum(axis=1)  # Count valid neighbors

        # Compute average velocity (avoid division by zero)
        n[n == 0] = np.inf
        vx[t][idx] = dx_sum / n
        vy[t][idx] = dy_sum / n

    # Calculate velocity magnitude
    mag = np.sqrt(vx**2 + vy**2)

    return vx, vy, mag


def visualize_wave(image,
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
    """Visualize wave propagation vectors on top of an image.

    This function creates a quiver plot showing wave direction and magnitude
    as arrows superimposed on the input image. The visualization can be filtered
    by magnitude threshold and downsampled for clarity.

    Args:
        image (np.ndarray): 2D background image, shape (height, width)
        idx (int): Time index to visualize (for multi-temporal data)
        wave_x (np.ndarray): 3D array of x-components of wave velocity, 
                            shape (time, height, width)
        wave_y (np.ndarray): 3D array of y-components of wave velocity,
                            shape (time, height, width)
        mag (np.ndarray, optional): Precomputed magnitude array. If None,
                                   will be calculated from wave_x and wave_y.
                                   shape (time, height, width)
        thre (float, optional): Magnitude threshold for displaying vectors.
                               If None, uses 25th percentile of magnitudes.
        step (int, optional): Spacing between arrows in pixels. Defaults to 10.
        linewidth (float, optional): Width of arrow shafts. Defaults to 0.5.
        head_width (float, optional): Width of arrow heads. Defaults to 1.5.
        head_length (float, optional): Length of arrow heads. Defaults to 2.
        save (bool or str, optional): If True, saves as PDF with default name.
                                     If string, uses as filename. Defaults to False.

    Returns:
        matplotlib.figure.Figure: The generated figure object
        matplotlib.axes.Axes: The axes containing the visualization
    """
    # Get image dimensions
    h, w = image.shape

    # Calculate magnitude if not provided
    if mag is None:
        mag = np.sqrt(wave_x**2 + wave_y**2)

    # Set default threshold if not provided
    if thre is None:
        thre = np.percentile(mag[idx],
                             25)  # Use 25th percentile as default threshold

    # Create figure and display background image
    fig, ax = plt.subplots()
    # ax.imshow(image, cmap='gray')  # Assuming grayscale image
    ax.imshow(image)  # Assuming grayscale image

    # Draw arrows on a grid with specified step size
    for i in range(0, h, step):
        for j in range(0, w, step):
            # Only draw arrows above magnitude threshold
            if mag[idx, i, j] > thre:
                ax.arrow(
                    j,
                    i,  # Arrow starting point (x,y)
                    wave_x[idx, i, j],  # x-component of vector
                    wave_y[idx, i, j],  # y-component of vector
                    fc='r',  # Face color (arrow head)
                    ec='r',  # Edge color (arrow shaft)
                    linewidth=linewidth,
                    head_width=head_width,
                    head_length=head_length,
                    length_includes_head=
                    False  # Head length not included in vector length
                )

    # Save figure if requested
    if save:
        save_name = f'../flow_{idx}.pdf' if isinstance(save, bool) else save
        plt.title(f'Wave propagation at t={idx}')
        plt.savefig(save_name, bbox_inches='tight', dpi=300)

    return fig, ax


class FileProcessor:
    '''
    The class is used for identifying the existing files or saving the new files. 
    '''

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
    '''
    This function is for making a flow video and save the result. 
    '''

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
    '''
    This function is for making a quiver video and save the result. 
    '''
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
    '''
    This function is for drawing the polar hist of velocity. 
    '''
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


# The belowed two functions is for making polar histgram videos
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
    '''Identify wave origin points from phase data based on specific temporal patterns.

    This function detects potential wave origins by analyzing phase evolution characteristics:
    - For 'iAdo2m' signals: Looks for points with negative-to-positive phase transitions
    - For 'jRGEC01a' signals: Looks for points with specific phase threshold crossings
    - Filters points based on duration and temporal characteristics

    Args:
        mod_phase (np.ndarray): 3D array of modified phase data, shape (time, height, width)
        sign (str): Signal type identifier ('iAdo2m' or 'jRGEC01a')
        fix (np.ndarray): 3D reference array for identifying valid points, shape (time, height, width)
        st (int, optional): Start time frame for analysis. Defaults to 1050.
        et (int, optional): End time frame for analysis. Defaults to 1450.
        hmin (int, optional): Minimum height (row) index for ROI. Defaults to 0.
        hmax (int, optional): Maximum height (row) index for ROI. Defaults to 359.
        wmin (int, optional): Minimum width (column) index for ROI. Defaults to 0.
        wmax (int, optional): Maximum width (column) index for ROI. Defaults to 359.
        recurrent_times (int, optional): Unused parameter (reserved for future functionality). Defaults to 15.
        top (int, optional): Number of top candidate points to return. Defaults to 10.
        continue_t_len_down (int, optional): Minimum duration threshold for valid phase evolution. Defaults to 100.
        continue_t_len_up (int, optional): Maximum duration threshold for valid phase evolution. Defaults to 1000.

    Returns:
        list: List of candidate origin points in global coordinates, each as [height, width]

    Raises:
        ValueError: If invalid signal type is specified
    '''

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
    """Cluster candidate origin points and select dominant region(s) based on spatial density.

    This function identifies coherent wave origin regions by clustering candidate points
    using either:
    1. KD-Tree based radius search (method='ckdtree')
    2. Hierarchical clustering (method='hierarchy')

    Args:
        origins (list or np.ndarray): Array of candidate origin points, shape (n_points, 2)
        radius (float, optional): Maximum distance for points to be considered clustered.
                                For 'ckdtree': connection radius between points
                                For 'hierarchy': maximum inter-cluster distance.
                                Defaults to 10.0.
        method (str, optional): Clustering method to use. Either 'ckdtree' or 'hierarchy'.
                              Defaults to 'hierarchy'.
        wave_num (int, optional): Number of wave origin regions to return. Only used when
                                method='hierarchy'. Defaults to 1.

    Returns:
        np.ndarray: For single wave_num: array of points in main cluster, shape (n_points_in_cluster, 2)
                   For multiple wave_num: tuple of arrays, each containing points for a cluster

    Raises:
        ValueError: If invalid clustering method is specified
    """
    if method == 'ckdtree':
        # Create KDTree for efficient radius searches
        kd_tree = cKDTree(np.asarray(origins))

        # Find all point pairs within specified radius
        pairs_within_radius = kd_tree.query_pairs(radius)

        # Collect all non-isolated points (those with neighbors within radius)
        non_isolated_points = set()
        for i, j in pairs_within_radius:
            non_isolated_points.add(tuple(origins[i]))
            non_isolated_points.add(tuple(origins[j]))

        return np.array(list(non_isolated_points))

    elif method == 'hierarchy':
        # Perform hierarchical clustering using complete linkage
        Z = linkage(origins, method='complete')

        # Form flat clusters based on distance threshold
        clusters = fcluster(Z, radius, criterion='distance')

        # Organize points by cluster label
        cluster_dict = {}
        for i, point in enumerate(origins):
            cluster_label = clusters[i]
            if cluster_label not in cluster_dict:
                cluster_dict[cluster_label] = []
            cluster_dict[cluster_label].append(point)

        # Sort clusters by size (number of points)
        sorted_dict = dict(
            sorted(cluster_dict.items(), key=lambda x: len(x[1]),
                   reverse=True))

        # Return requested number of largest clusters
        if wave_num == 1:
            return np.asarray(list(sorted_dict.values())[0])
        else:
            return tuple(
                np.asarray(list(sorted_dict.values())[i])
                for i in range(wave_num))

    else:
        raise ValueError(f"Invalid clustering method: {method}. "
                         "Please use either 'ckdtree' or 'hierarchy'.")


def min_circle(points):
    """Compute the minimum enclosing circle for a set of 2D points using a simple iterative algorithm.

    This function implements a basic algorithm to find the smallest circle that encloses all given points.
    The algorithm works by iteratively expanding the circle to include points that fall outside it.

    Args:
        points (list or np.ndarray): Collection of 2D points to enclose, where each point is 
                                    represented as [x, y]. Shape should be (n_points, 2).

    Returns:
        tuple: A 3-element tuple representing the minimum enclosing circle:
            - center_x (float): x-coordinate of circle center
            - center_y (float): y-coordinate of circle center
            - radius (float): radius of the circle

    Note:
        This implementation provides an approximate solution and may not always find the true 
        minimum enclosing circle. For exact solutions, consider using more sophisticated 
        algorithms like Welzl's algorithm.
    """

    def is_inside(circle, point):
        """Check if a point lies inside or on the boundary of a given circle.

        Args:
            circle (tuple): Circle represented as (center_x, center_y, radius)
            point (list or tuple): Point coordinates [x, y]

        Returns:
            bool: True if point is inside/on circle, False otherwise
        """
        distance = math.hypot(circle[0] - point[0], circle[1] - point[1])
        return distance <= circle[
            2] + 1e-6  # Small tolerance for floating point precision

    def update_circle(circle, point):
        """Update the circle to include a new point on its boundary.

        The new circle will have the point and the previous circle's boundary point as diameter.

        Args:
            circle (tuple): Current circle (center_x, center_y, radius)
            point (list or tuple): New point to include [x, y]

        Returns:
            tuple: Updated circle (center_x, center_y, radius)
        """
        center_x = (circle[0] + point[0]) / 2
        center_y = (circle[1] + point[1]) / 2
        radius = math.hypot(center_x - point[0], center_y - point[1])
        return (center_x, center_y, radius)

    # Initialize with first point (zero-radius circle)
    random_point = random.choice(points)
    min_circle = (random_point[0], random_point[1], 0.0)

    # Iteratively expand the circle to include all points
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
    """Visualize wave propagation patterns with origin regions and propagation vectors.

    This function creates a comprehensive visualization showing:
    1. The background activity pattern
    2. Wave propagation vectors (direction and magnitude)
    3. Identified origin regions
    4. Wave propagation patterns from origins

    Args:
        image (np.ndarray): 3D array of background activity images, shape (time, height, width)
        phase (np.ndarray): 3D array of phase data, shape (time, height, width)
        idx (int): Time index to visualize (relative to start of analysis window)
        wave_x (np.ndarray): 3D array of x-direction wave velocities, shape (time, height, width)
        wave_y (np.ndarray): 3D array of y-direction wave velocities, shape (time, height, width)
        origin_region (list or np.ndarray): Detected origin points (output from select_origin_region)
        wave_num (int): Number of distinct wave origins to visualize
        title_name (str, optional): Title for the plot. Defaults to None.
        mag (np.ndarray, optional): Precomputed magnitude of wave vectors. Defaults to None.
        thre (float, optional): Threshold for displaying wave vectors. Defaults to 3.
        step (int, optional): Spacing between displayed vectors. Defaults to 10.
        phase_step (int, optional): Time delta for phase propagation analysis. Defaults to 10.
        linewidth (float, optional): Width of vector lines. Defaults to 1.0.
        head_width (float, optional): Width of vector heads. Defaults to 1.5.
        head_length (float, optional): Length of vector heads. Defaults to 2.
        save (bool, optional): Whether to save the figure. Defaults to False.
        quiver_thre (float, optional): Phase difference threshold for propagation vectors. Defaults to 0.01.

    Returns:
        None: Displays or saves the visualization
    """
    # Initialize figure and prepare data
    t, h, w = image.shape

    # Calculate magnitude if not provided
    if mag is None:
        mag = np.sqrt(wave_x**2 + wave_y**2)

    # Set default threshold if not provided
    if thre is None:
        thre = np.percentile(mag[idx], 25)

    # Create figure and display background
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

    # Plot wave propagation vectors (quiver plot)
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

    # Handle single wave origin case
    if wave_num == 1:
        Points = origin_region.copy()
        Points[:, [0, 1]] = Points[:, [1,
                                       0]]  # Swap x,y coordinates for plotting
        center_x, center_y, radius = min_circle(Points)
        center_x = int(center_x)
        center_y = int(center_y)

        # Plot origin points and region
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

        # Find phase propagation endpoints
        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)

        # Plot propagation vectors from origin
        for e in range(0, len(endpoints)):
            u = (endpoints[e, 0] -
                 center_x) / 4  # Scale vectors for visibility
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

    # Handle multiple wave origins case
    elif wave_num > 1:
        Center_x = np.zeros(wave_num)
        Center_y = np.zeros(wave_num)

        # Process each origin region
        for i in range(wave_num):
            Points = origin_region[i].copy()
            Points[:, [0, 1]] = Points[:, [1, 0]]  # Swap x,y coordinates
            center_x, center_y, radius = min_circle(Points)
            Center_x[i] = int(center_x)
            Center_y[i] = int(center_y)

            # Plot origin points
            ax.scatter(Points[:, 0],
                       Points[:, 1],
                       color='red',
                       label='Origins')

        # Calculate and plot mean origin region
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

        # Find and plot propagation vectors
        endpoints = np.argwhere(
            abs(phase[idx + phase_step] -
                phase[idx, center_y, center_x]) < quiver_thre)
        for e in range(0, len(endpoints)):
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

    # Finalize and display/save plot
    if save and title_name:
        plt.title(f'{title_name} flow at origin t{1050+idx}')
    plt.show()


#
def vis_wave_local(image,
                   phase,
                   idx,
                   wave_x,
                   wave_y,
                   origin_region,
                   wave_num=1,
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
    '''
    This function shares the same idea with 'vis_wave', but focus on local area. 
    '''
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
    ax.set_xlim([200, 320])
    ax.set_ylim([180, 60])
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
    '''
    This function shares the same idea with 'vis_wave', but the center points are aligned. 
    '''
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
    '''
    This function shares the same idea with 'vis_wave_refine', but focus on local area. 
    '''
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
    '''
    This function shares the same idea with 'vis_wave', but uses standard directions. 
    '''
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
    ax.set_xlim([200, 320])
    ax.set_ylim([180, 60])
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
    """Find the vector from a list that has the smallest angle with a reference vector,
    and return its magnitude.

    This function calculates the angle between a reference vector and each vector in a list,
    identifies the vector with the smallest angle, and returns its magnitude (L2 norm).

    Args:
        standard_vec (list or np.ndarray): Reference vector for angle comparison, shape (n_dimensions,)
        vec_list (list of lists or np.ndarray): List of vectors to compare against, 
                                              shape (n_vectors, n_dimensions)

    Returns:
        float: The magnitude (L2 norm) of the vector from vec_list that has the smallest angle 
              with standard_vec

    Note:
        - All vectors should be of the same dimensionality
        - For identical angles, returns the magnitude of the first occurrence
        - Uses the dot product formula: cosθ = (a·b)/(||a||·||b||)
    """
    angle = []

    # Calculate angle between standard_vec and each vector in vec_list
    for i in range(len(vec_list)):
        # Compute dot product magnitude (||a||·||b||)
        l = np.sqrt(np.asarray(standard_vec).dot(np.asarray(standard_vec))) * \
            np.sqrt(np.asarray(vec_list[i]).dot(np.asarray(vec_list[i])))

        # Handle potential division by zero for zero vectors
        if l == 0:
            cos = 0
        else:
            # Compute cosine of angle between vectors
            cos = np.asarray(standard_vec).dot(np.asarray(vec_list[i])) / l

        # Clip cosine value to avoid numerical errors in arccos
        cos = np.clip(cos, -1.0, 1.0)
        angle.append(np.arccos(cos))

    # Find vector with minimum angle and return its magnitude
    min_angle_idx = np.argmin(np.asarray(angle))
    return np.sqrt(
        np.asarray(vec_list[min_angle_idx]).dot(
            np.asarray(vec_list[min_angle_idx])))


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
    '''
    This function shares the same idea with 'vis_wave', but uses standard arrows. 
    '''
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
    ax.set_xlim([200, 320])
    ax.set_ylim([180, 60])
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
