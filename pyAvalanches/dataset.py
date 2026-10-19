from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Union, Sequence, Tuple
import numpy as np
import mne
from scipy.stats import zscore, norm
import matplotlib.pyplot as plt


class Dataset:
    """
    Stores multi-trial EEG/neurophysiological time series data along with
    associated metadata such as sampling frequency, filter settings,
    channel names, and trial information.

    Parameters
    ----------
    data : np.ndarray
        Neural time series data of shape (n_trials, n_channels, n_time_points) or (n_channels, n_time_points).
    samp_freq : float
        Sampling frequency in Hz.
    fmin : float
        Lower frequency in Hz.
    fmax : float
        Higher frequency in Hz.
    channel_names : list of int or str, optional
        Channel names or indices. Defaults to None.
    trial_labels : list of int or str, optional
        Condition label for each trial. Defaults to None.
    t_max : float, optional
        Trial duration in seconds. Defaults to None.
    info : dict, optional
        Additional dataset metadata. Defaults to None.
    """

    def __init__(
            self,
            data: np.ndarray,
            samp_freq: float,
            fmin: float,
            fmax: float,
            channel_names: Optional[List[Union[int, str]]] = None,
            #trial_ids: Optional[List[int]] = None,
            trial_labels: Optional[List[Union[int, str]]] = None,
            t_max: Optional[float] = None,
            data_zscored = None,
            info: Optional[dict] = None
    ):
        self.data = data.copy()  # shape: (n_trials, n_channels, n_time_points)
        self.samp_freq = samp_freq
        self.fmin = fmin
        self.fmax = fmax
        self.channel_names = channel_names
        self.trial_ids = list(range(self.data.shape[0])) # set by default
        self.trial_labels = trial_labels
        self.t_max = t_max
        self.data_zscored = None
        self.info = info

    #def __post_init__(self):

        if self.data.ndim not in [2,3]:
            raise ValueError("Data must be 2D (channels x time) or 3D (trials x channels x time)")

        if self.data.ndim == 2:
            self.data = np.expand_dims(self.data, axis=0)

        n_trials = self.data.shape[0]

        # t_max truncation
        if self.t_max is not None:
            if self.t_max <= 0:
                raise ValueError(f"t_max={self.t_max} must be a positive integer number of time points.")
            if self.t_max > self.data.shape[-1]:
                raise ValueError(
                    f"t_max={self.t_max} exceeds the available number of time points ({self.data.shape[-1]})."
                )
            self.data = self.data[..., :self.t_max]

        # check if the number of labels matches the number of trials
        if self.trial_labels is not None and len(self.trial_labels) != n_trials:
            raise ValueError("trial_labels length must match number of trials")

        # check if the number of labels matches the number of trials
        #if self.trial_ids is not None and len(self.trial_ids) != n_trials:
        #    raise ValueError("trial_ids length must match number of trials")

        # automatically generate trial identifiers if none are given
        if self.trial_ids is None:
            self.trial_ids = list(range(self.data.shape[0]))

        if self.channel_names is not None:
            if len(self.channel_names) != self.data.shape[1]:
                raise ValueError("channel names length must match number of channels")

    @property
    def n_trials(self):
        """
        Number of trials in the dataset.

        Returns
        -------
        int
            The number of trials (first dimension of the data array).
        """
        return self.data.shape[0]

    @property
    def n_channels(self):
        """
        Number of channels or ROIs in the dataset.

        Returns
        -------
        int
            The number of channels or ROIs (second dimension of the data array).
        """
        return self.data.shape[1]

    @property
    def n_time_points(self):
        """
        Number of time points in the dataset or in a single trial.

        Returns
        -------
        int
            The number of time points (third dimension of the data array).
        """
        return self.data.shape[2]

    def get_trial(self, trial_index: Union[int,List[int]]):
        """
        Retrieve the data for one or more trials by index.

        Parameters
        ----------
        trial_index : int or list of int
            Index or list of indices of the trial(s) to retrieve.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(channels, time_points)`` if `trial_index` is an
            int, or ``(n_trials, channels, time_points)`` if `trial_index` is a
            list of int.
        """
        return self.data[trial_index, :, :].copy()

    def get_channel(self, channel_index: int):
        """
        Retrieve the data for a single channel across all trials and time points.

        Parameters
        ----------
        channel_index : int
            Index of the channel to retrieve.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_trials, time_points)`` containing the data for
            the specified channel.
        """
        return self.data[:, channel_index, :].copy()

    def get_time_points(self, time_index: int):
        """
        Retrieve the data at a specific time point across all trials and channels.

        Parameters
        ----------
        time_index : int
            Index of the time point to retrieve.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(n_trials, channels)`` containing the data at the
            specified time point.
        """
        return self.data[:, :, time_index].copy()

    def zscore_data(self, by_trial: bool):
        """
        Z-scores the dataset using one of two normalization strategies and
        store the results in self.data_zscored.

        If ``by_trial=True``, each trial is z-scored independently across its
        channels and time points. If ``by_trial=False``, each channel is
        z-scored globally by pooling all trials and time points together before
        computing the mean and standard deviation.

        Parameters
        ----------
        by_trial : bool
            If True, z-scoring is applied independently to each trial.
            If False, z-scoring is applied per channel across all trials
            and time points combined.
        """

        if by_trial:
            zscored_data_ep = np.zeros((self.n_trials, self.n_channels, self.n_time_points))
            for i in range(self.n_trials):
                zscored_trial = zscore(self.data[i,:,:], axis=1)
                zscored_data_ep[i,:,:] = zscored_trial
        else:
            # move channels to axis 0
            temp = np.transpose(self.data, (1, 0, 2))
            # flatten time and epochs
            temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
            # z scoring
            zscored_data = zscore(temp_nc, axis=1)
            # reshape back
            temp_zscored_data_ep = np.reshape(zscored_data, (np.shape(temp)[0], np.shape(temp)[1], np.shape(temp)[2]))
            zscored_data_ep = np.transpose(temp_zscored_data_ep, (1, 0, 2))

        # replace original data with zscored ones
        self.data_zscored = zscored_data_ep

    def _get_mean_and_std(self):
        """Compute global mean and standard deviation of z-scored neural activity.

        Computes statistics across all channels, trials, and time points by:
        1. Transposing data to move channels to the first axis
        2. Flattening trials and time dimensions
        3. Z-scoring across channels
        4. Computing mean absolute value and standard deviation

        These statistics are typically used for threshold-based binarization
        of neural data (see `_threshold_data_std`).

        Returns
        -------
        mean_signal : float
            Global mean of absolute z-scored activity, rounded to 1 decimal place.
        std_signal : float
            Global standard deviation of absolute z-scored activity, rounded to
            1 decimal place.

        Notes
        -----
        - Z-scoring is performed per channel (axis=1) after data reshaping
        - The absolute value is computed before computing mean/std to capture
          activity magnitude regardless of sign
        - Averages are computed across channels to produce global statistics
        - Results are rounded to 1 decimal place for interpretability

        Assumes
        --------
        self.data : np.ndarray
            Data array of shape (n_trials, n_channels, n_time_bins) or
            similar with trials/epochs in axis 0, channels in axis 1,
            and time in axis 2.
        """
        # move channels to axis 0
        temp = np.transpose(self.data, (1, 0, 2))
        # flatten time and epochs
        temp_nc = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))
        # z scoring
        zscored_data = zscore(temp_nc, axis=1)
        # compute mean and std
        mean_signal = round(float(np.mean(np.mean(np.abs(zscored_data), axis=0))), 1)
        std_signal = round(float(np.mean(np.std(np.abs(zscored_data), axis=0))), 1)

        return mean_signal, std_signal

    def _filter_data(self, s_freq, l_freq=None, h_freq=None, **kwargs):
        """
        Filters the dataset in-place using MNE's filter_data function.

        Parameters
        ----------
        s_freq : float
            Sampling frequency of the data in Hz.
        l_freq : float, optional
            Lower cutoff frequency for highpass/bandpass filtering in Hz.
            If None, no highpass filter is applied. Defaults to None.
        h_freq : float, optional
            Upper cutoff frequency for lowpass/bandpass filtering in Hz.
            If None, no lowpass filter is applied. Defaults to None.
        **kwargs
            Additional keyword arguments passed to `mne.filter.filter_data`.

        Warning
        ---------
        The original data in `self.data` is overwritten with the filtered
        values, and `self.l_freq` and `self.h_freq` are updated to reflect
        the applied filter settings.
        """

        data_to_filter = self.data
        filtered_data = mne.filter.filter_data(
            data=data_to_filter,
            sfreq=s_freq,
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=True,
            **kwargs)

        self.data = filtered_data
        if l_freq is not None:
            self.fmax = l_freq
        if h_freq is not None:
            self.fmin = h_freq

    def get_binarized_data(self, threshold_method, threshold):
        """
        Binarize the data by applying a threshold.

        Values whose absolute magnitude exceeds the threshold are set to 1;
        all others are set to 0. A warning is printed if no values exceed the
        threshold.

        Parameters
        ----------
        threshold_method : {'fixed', 'std'}
            Method used to determine the threshold:

            - ``'fixed'`` : Use `threshold` directly as the absolute cutoff
              value.
            - ``'std'`` : Use `threshold` as a multiplier of the standard
              deviation of the data (i.e., the effective cutoff is
              ``threshold * data.std()``).

        threshold : float
            Threshold value. Interpreted as an absolute value when
            `threshold_method` is ``'fixed'``, or as the number of standard
            deviations when `threshold_method` is ``'std'``.

        Returns
        -------
        numpy.ndarray
            Binary array with the same shape as ``self.data``, where 1
            indicates that the absolute value at that position exceeds the
            threshold and 0 otherwise.

        Raises
        ------
        ValueError
            If `threshold_method` is not ``'fixed'`` or ``'std'``.
        """

        if self.data_zscored is not None:

            # if fixed
            if threshold_method == 'fixed':
                binarized_data = np.where(np.abs(self.data_zscored) > threshold, 1, 0)
                if not np.any(binarized_data):  # check if all elements are zero -> no signal above threshold gives a warning
                    print(f'WARNING: No signal found above threshold {threshold}')

                return binarized_data

            # if std
            elif threshold_method == 'std':
                #std = self.data.std()
                mean_signal, std_signal = self._get_mean_and_std()
                binarized_data = np.where(
                                    np.abs(self.data_zscored) > (mean_signal + threshold * std_signal), 1, 0
                                )
                if not np.any(binarized_data):  # check if all elements are zero -> no signal above threshold gives a warning
                    print(f'WARNING: No signal found above threshold {threshold}')

                return binarized_data

            else:
                raise ValueError("No valid thresholding method was given'")

        else:
            raise ValueError("The dataset needs to be z_scored before performing this action")

def create_dataset_from_mne(mne_data: Union[mne.io.BaseRaw, mne.BaseEpochs], t_max: float = None) -> Dataset :
    """
    Creates a Dataset object from an MNE Raw or Epochs object.
    For Raw objects, the data is treated as a single trial.
    For Epochs objects, each epoch is treated as a separate trial with
    its corresponding event label.

    Parameters
    ----------
    mne_data : mne.io.BaseRaw or mne.BaseEpochs
        The MNE object to convert. Must be either a Raw or Epochs
        instance.
    t_max : float, optional
        Trial duration cutoff in time points, if needed. Defaults to None.

    Returns
    -------
    Dataset
        A Dataset object containing the data and metadata extracted
        from the MNE object.

    Raises
    ------
    TypeError
        If `mne_data` is not an instance of `mne.io.BaseRaw` or
        `mne.BaseEpochs`.

    Notes
    -----
    For Raw objects, the data is expanded to shape (1, n_channels,
    n_time_points) to match the expected (n_trials, n_channels,
    n_time_points) format. Trial labels are only set for Epochs objects,
    using the event codes from `mne_data.events`.
    """

    if isinstance(mne_data, mne.io.BaseRaw):
        eeg_raw_data = mne_data.get_data()
        # add the trial dimension = 1
        eeg_raw_data = np.expand_dims(eeg_raw_data, axis=0)
        ch_names = mne_data.ch_names
        trial_ids = [0]
        #trial_labels = [0]

    elif isinstance(mne_data, mne.BaseEpochs):
        eeg_raw_data = mne_data.get_data()
        ch_names = mne_data.ch_names
        trial_ids = [i for i in range(len(mne_data))]
        trial_labels = mne_data.events[:, -1]

    else:
        raise TypeError("Input must be Raw or Epochs object")

    dataset_out = Dataset(data=eeg_raw_data,
                           samp_freq=mne_data.info["sfreq"],
                           fmax=mne_data.info["lowpass"],
                           fmin=mne_data.info["highpass"],
                           channel_names=ch_names,
                           #trial_ids=trial_ids,
                           t_max = t_max,
                           )

    if isinstance(mne_data, mne.BaseEpochs):
        dataset_out.trial_labels = trial_labels

    return dataset_out


def _build_dataset(input_data: Union[mne.io.BaseRaw, mne.BaseEpochs, dict]):
    """
    Converts any supported input into a Dataset.

    Parameters
    ----------
    input_data : mne.io.BaseRaw, mne.BaseEpochs or dict
        A dict must contain the keys 'data_array', 'samp_freq', 'fmin',
        'fmax', 'channel_names', 'trial_labels' and 't_max'.

    Returns
    -------
    Dataset

    Raises
    ------
    TypeError
        If the input is of an unsupported type.
    KeyError
        If a dict input is missing a required key.
    """
    # BaseRaw and BaseEpochs are the abstract parents, so this accepts every
    # concrete MNE class (Raw, RawArray, Epochs, EpochsArray, EpochsFIF, ...).
    if isinstance(input_data, (mne.io.BaseRaw, mne.BaseEpochs)):
        return create_dataset_from_mne(input_data)

    if isinstance(input_data, dict):
        required = {"data_array", "samp_freq", "fmin", "fmax",
                    "channel_names", "trial_labels", "t_max"}
        # Report ALL missing keys at once instead of failing on the first.
        missing = required - input_data.keys()
        if missing:
            raise KeyError(
                f"input dict is missing required key(s): {sorted(missing)}"
            )

        return Dataset(
            data=input_data["data_array"],
            samp_freq=input_data["samp_freq"],
            fmin=input_data["fmin"],
            fmax=input_data["fmax"],
            channel_names=input_data["channel_names"],
            trial_labels=input_data["trial_labels"],
            t_max=input_data["t_max"],
        )

    raise TypeError(
        f"input_data must be an mne Raw object, an mne Epochs object, or a "
        f"dict, got {type(input_data).__name__}"
    )


#def plot_amplitude_distribution(dataset: Dataset):
def plot_amplitude_distribution(data: Union[mne.io.BaseRaw, mne.BaseEpochs, dict]):
    """
    Plots the amplitude distribution of each channel in the dataset on a
    semi-logarithmic scale, overlaid with the channel mean and a standard
    normal distribution for comparison.

    Parameters
    ----------
    data : mne.io.BaseRaw, mne.BaseEpochs or dict
        A dict must contain the keys 'data_array', 'samp_freq', 'fmin',
        'fmax', 'channel_names', 'trial_labels' and 't_max'.

    Notes
    -----
    Each channel's amplitude distribution is plotted as a semi-logarithmic
    line, allowing visual assessment of how closely the data follows a
    Gaussian distribution.
    """

    """
    if isinstance(data, mne.io.BaseRaw):
        dataset = create_dataset_from_mne(data)

    elif isinstance(data, mne.Epochs):
        dataset = create_dataset_from_mne(data)

    elif isinstance(data, dict):
        dataset = Dataset(
            data=data["data_array"],
            samp_freq=data["samp_freq"],
            fmin=data["fmin"],
            fmax=data["fmax"],
            channel_names=data["channel_names"],
            # trial_ids=info["trial_ids"],
            trial_labels=data["trial_labels"],
            t_max=data["t_max"],
        )

    #elif isinstance(data, np.ndarray) and info is None:
    #    raise TypeError("When a np.ndarray is provided, an info dictionary is needed")
    else:
        raise TypeError("Input must be Raw or Epochs object or np.ndarray")
    """

    dataset = _build_dataset(data)

    #TODO check how this should be z-scored
    dataset.zscore_data(by_trial=False)

    NN = dataset.n_channels
    # move channels to axis 0
    trans_data = np.transpose(dataset.data_zscored, (1, 0, 2))
    # flatten time and epochs
    flattened_data = np.reshape(trans_data, (np.shape(trans_data)[0], np.shape(trans_data)[1] * np.shape(trans_data)[2]))

    xmin = flattened_data.min()
    xmax = flattened_data.max()
    nbin = 100
    bin_edges = np.linspace(xmin, xmax, nbin + 1)
    xx = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    histog = np.zeros((NN, nbin))

    # make histograms of amplitudes
    for k in range(NN):
        node_data = flattened_data[k,:]
        # normalized histogram
        counts, _ = np.histogram(node_data, bins=bin_edges, density=True)
        histog[k, :] = counts

    histo_mean = histog.mean(axis=0)

    # gaussian distibution
    y = np.arange(-5, 5.0 + 0.1, 0.1)
    f = norm.pdf(y, loc=0.0, scale=1.0)

    # log plot
    plt.figure(figsize=(8, 6))
    for i in range(NN):
        plt.semilogy(xx, histog[i, :], '-*', markersize=2, linewidth=0.7, alpha=0.35)
    plt.semilogy(xx, histo_mean, 'm-', linewidth=2.5, label='channel mean')
    plt.semilogy(y, f, 'k-', linewidth=2.5, label='N(0,1)')
    plt.xlim([-6, 6])
    plt.ylim([8e-6, 1])
    plt.xlabel('Amplitude [SD]')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Average Amplitude Distribution (by channels, semilogy) with Gaussian Fit')
    plt.tight_layout()
    plt.show()