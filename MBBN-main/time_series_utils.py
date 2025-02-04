from scipy.signal import butter, filtfilt # added
import numpy as np

# Calculate the repetition time (TR) depending on the site
def repetition_time(site):

    if 'Amsterdam-AMC' in site:
        TR = 2.375
    elif 'Amsterdam-VUmc' in site:
        TR = 1.8
    elif 'Barcelona-HCPB' in site:
        TR = 2
    elif 'Bergen' in site:
        TR = 1.8
    elif 'Braga-UMinho-Braga-1.5T' in site:
        TR = 2
    elif 'Braga-UMinho-Braga-1.5T-act' in site:
        TR = 2
    elif 'Braga-UMinho-Braga-3T' in site:
        TR = 1
    elif 'Brazil' in site:
        TR = 2
    elif 'Cape-Town-UCT-Allegra' in site:
        TR = 1.6
    elif 'Cape-Town-UCT-Skyra' in site:
        TR = 1.73
    elif 'Chiba-CHB' in site:
        TR = 2.3
    elif 'Chiba-CHBC' in site:
        TR = 2.3 
    elif 'Chiba-CHBSRPB' in site:
        TR = 2.5 
    elif 'Dresden' in site:
        TR = 0.8 
    elif 'Kyoto-KPU-Kyoto1.5T' in site:
        TR = 2.411 
    elif 'Kyoto-KPU-Kyoto3T' in site:
        TR = 2
    elif 'Kyushu' in site:
        TR = 2.5
    elif 'Milan-HSR' in site:
        TR = 2
    elif 'New-York' in site:
        TR = 1
    elif 'NYSPI-Columbia-Adults' in site:
        TR = 0.85
    elif 'NYSPI-Columbia-Pediatric' in site:
        TR = 0.85
    elif 'Yale-Pittinger-HCP-Prisma' in site:
        TR = 0.8
    elif 'Yale-Pittinger-HCP-Trio' in site:
        TR = 0.7
    elif 'Yale-Pittinger-Yale-2014' in site:
        TR = 2
    elif 'Bangalore-NIMHANS' in site:
        TR = 2 
    elif 'Barcelone-Bellvitge-ANTIGA-1.5T' in site:
        TR = 2
    elif 'Barcelone-Bellvitge-COMPULSE-3T' in site:
        TR = 2
    elif 'Barcelone-Bellvitge-PROV-1.5T' in site:
        TR = 2
    elif 'Barcelone-Bellvitge-RESP-CBT-3T' in site:
        TR = 2
    elif 'Seoul-SNU' in site:
        TR = 3.5
    elif 'Shanghai-SMCH' in site:
        TR = 3
    elif 'UCLA' in site:
        TR = 2
    elif 'Vancouver-BCCHR' in site:
        TR = 2
    elif 'Yale-Gruner' in site:
        TR = 2
    else:
        raise ValueError(f"Site '{site}' does not have a defined TR value in TR_mappings. Please add it.")

    return TR


def compute_imf_bandwidths(u, fs, threshold=0.05):
    """
    Compute the bandwidths of IMFs using the Fourier spectrum directly from VMD output.
    
    This version correctly extracts frequency bounds in Hz, avoiding the issue of 
    symmetric zero-centered results.
    
    Parameters:
    u (ndarray): IMFs (shape: K x N, where K is the number of IMFs, N is the time samples).
    fs (float): Sampling frequency of the time series (Hz).
    threshold (float): Power threshold for frequency support (default 1% of max power).

    Returns:
    dict: Band cutoffs in Hz as { 'imf1_lb': ..., 'imf1_hb': ..., ... }
    """
    K, N = u.shape  # Number of IMFs and time samples
    f_N = fs / 2  # Nyquist frequency
    freqs = np.fft.fftfreq(N, d=1/fs)  # Compute frequencies WITHOUT shifting
    positive_freqs = freqs[:N//2]  # Keep only positive frequencies
    band_cutoffs = {}

    for k in range(K):
        # Compute the Fourier Transform of the IMF
        U_k = np.fft.fft(u[k, :])
        power_spectrum = np.abs(U_k) ** 2

        # Normalize power and apply threshold
        power_threshold = threshold * np.max(power_spectrum)
        
        # Extract frequency support only from the positive range
        freq_support = positive_freqs[power_spectrum[:N//2] > power_threshold]

        if len(freq_support) > 0:
            f_min = np.min(freq_support)  # Minimum frequency with significant power
            f_max = np.max(freq_support)  # Maximum frequency with significant power
        else:
            f_min, f_max = 0, 0  # In case no significant power is detected

        # Store the frequency cutoffs
        band_cutoffs[f'imf{k+1}_lb'] = max(0, f_min)  # Ensure non-negative frequencies
        band_cutoffs[f'imf{k+1}_hb'] = min(f_N, f_max)  # Ensure does not exceed Nyquist

    return band_cutoffs


def bandpass_filter_2d(data, lowcut, highcut, fs, order=4):
    """
    Applies a Butterworth bandpass filter to each ROI in a 2D time-series dataset.
    
    Parameters:
    - data: numpy array of shape (#ROIs, #timepoints), where each row is a time series for one ROI.
    - lowcut: Lower cutoff frequency (Hz).
    - highcut: Upper cutoff frequency (Hz).
    - fs: Sampling frequency (Hz) = 1 / TR.
    - order: Order of the Butterworth filter (default = 4).

    Returns:
    - filtered_data: numpy array of the same shape as 'data' with filtered time series.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply filter to each ROI (row-wise)
    filtered_data = np.array([filtfilt(b, a, roi_signal) for roi_signal in data])

    return filtered_data