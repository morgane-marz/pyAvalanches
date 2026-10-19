from dataclasses import dataclass, field
from typing import Union, List, Optional, Tuple, Dict, Sequence
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import mne
from scipy.stats import zscore
from sklearn.base import BaseEstimator, TransformerMixin
from .dataset import Dataset, create_dataset_from_mne, _build_dataset

class AvalancheObj:
    """
    Stores the results of avalanche detection for a single trial.

    Parameters
    ----------
    trial_id : int
        Unique identifier for the trial.
    avalanches_mat : list of np.ndarray
        List of arrays representing the binarized spatiotemporal activity
        matrix for each detected avalanche.
    time_bins : int
        Number of time bins used for optional time subsampling.
    sizes : list of int, optional
        Number of activations in each avalanche. Defaults to None.
    durations : list of int, optional
        Duration in time bins of each avalanche. Defaults to None.
    patterns : list of np.ndarray, optional
        Spatial activation patterns for each avalanche. Defaults to None.
    patterns_temporal : list of np.ndarray, optional
        Temporal activation patterns for each avalanche. Defaults to None.
    bounds : list of tuple, optional
        Start and end time point indices for each avalanche. Defaults to None.
    flexibility : int, optional
        Number of distinct spatial patterns observed across avalanches.
        Defaults to None.
    avalanche_count : int, optional
        Total number of detected avalanches in the trial. Defaults to None.
    trial_label : int or str, optional
        Condition label for the trial (e.g., 'MI', 'rest'). Defaults to None.
    """

    def __init__(
            self,
            trial_id: int,
            avalanches_mat: List[np.ndarray],
            time_bins: Optional[int] = 1,
            sizes: Optional[List[int]] = None,
            durations: Optional[List[int]] = None,
            patterns: Optional[List[np.array]] = None,
            patterns_temporal: Optional[List[np.array]] = None,
            bounds: Optional[List[tuple]] = None,
            flexibility: Optional[int] = None,
            avalanche_count: Optional[int] = None,
            trial_label: Optional[Union[int, str]] = None
    ):
        self.trial_id = trial_id
        self.avalanches_mat = avalanches_mat
        self.time_bins = time_bins
        self.sizes = sizes
        self.durations = durations
        self.patterns = patterns
        self.patterns_temporal = patterns_temporal
        self.bounds = bounds
        self.flexibility = flexibility
        self.avalanche_count = avalanche_count
        self.trial_label = trial_label  # could be MI, rest ...

    def __repr__(self):
        return (
            f"AvalancheObj(trial_id={self.trial_id}, "
            f"n_avalanches={self.avalanche_count})"
        )


class TrialAvalanches:
    """
    Stores a collection of AvalancheObj instances, one per trial.

    Parameters
    ----------
    trials : list of AvalancheObj
        List of avalanche detection results, one entry per trial.
        Defaults to an empty list.
    """

    def __init__(
            self,
            trials: Optional[List[AvalancheObj]] = None
    ):
        self.trials = trials if trials is not None else []

    def __repr__(self):
        return (
            f"TrialAvalanches(n_trials={len(self.trials)})"
        )

    @property
    def trial_labels(self):
        """Get condition labels for all trials in the collection.

        Returns
        -------
        list
            List of condition labels for all trials. Each element corresponds
            to the trial_label attribute of an AvalancheObj trial.
        """
        all_trials_labels = [trial.trial_label for trial in self.trials]
        return all_trials_labels

    @property
    def trial_indexes(self):
        """Get trial IDs for all trials in the collection.

        Returns
        -------
        list
            List of trial IDs for all trials. Each element corresponds
            to the trial_id attribute of an AvalancheObj trial.
        """
        all_trials_ids = [trial.trial_id for trial in self.trials]
        return all_trials_ids

    @property
    def trial_count(self):
        """Get the total number of trials stored in the collection.

        Returns
        -------
        int
            Total number of trials currently stored.
        """
        return len(self.trials)

    def add_trial(self, trial: AvalancheObj):
        """Add an AvalancheObj trial result to the collection.

        Parameters
        ----------
        trial : AvalancheObj
            The trial object to be added to the trials collection.
        """
        self.trials.append(trial)

    def get_trial_data(self, trial_idx):
        """
        Retrieves the AvalancheObj for a specific trial by its ID.

        Parameters
        ----------
        trial_idx : int
            The trial ID to retrieve.

        Returns
        -------
        AvalancheObj
            The avalanche data object corresponding to the given trial ID.

        Raises
        ------
        ValueError
            If the trial_idx is not found in the trials list.
        """

        for trial in self.trials:
            if int(trial.trial_id) == int(trial_idx):
                return trial

        available_ids = [trial.trial_id for trial in self.trials]
        raise ValueError(
            f"Trial ID '{trial_idx}' not found. "
            f"Available trial IDs are: {available_ids}"
        )

    def get_all_durations(self, merged: bool) -> List:
        """
        Retrieves avalanche durations across all trials.

        Parameters
        ----------
        merged : bool
            If True, returns a single flat list of all durations across trials.
            If False, returns a list of lists, one per trial.

        Returns
        -------
        list
            Avalanche durations across all trials, either merged or nested.
        """

        durations = []
        for trial in self.trials:
            if merged:
                durations = durations + trial.durations
            else:
                durations.append(trial.durations)
        return durations

    def get_durations_by_cond(self, cond, merged: bool) -> List:
        """
        Retrieves avalanche durations for all trials matching a given condition.

        Parameters
        ----------
        cond : int or str
            The condition label used to filter trials.
        merged : bool
            If True, returns a single flat list of all durations across matching
            trials. If False, returns a list of lists, one per matching trial.

        Returns
        -------
        list
            Avalanche durations for trials matching the specified condition,
            either merged or nested.
        """
        durations = []
        for trial in self.trials:
            if trial.trial_label == cond:
                if merged:
                    durations = durations + trial.durations
                else:
                    durations.append(trial.durations)
        return durations

    def get_all_sizes(self, merged: bool) -> List:
        """
        Retrieves avalanche sizes across all trials.

        Parameters
        ----------
        merged : bool
            If True, returns a single flat list of all sizes across trials.
            If False, returns a list of lists, one per trial.

        Returns
        -------
        list
            Avalanche sizes across all trials, either merged or nested.
        """
        sizes = []
        for trial in self.trials:
            if merged:
                sizes = sizes + trial.sizes
            else:
                sizes.append(trial.sizes)
        return sizes

    def get_sizes_by_cond(self, cond, merged: bool) -> List:
        """
        Retrieves avalanche sizes for all trials matching a given condition.

        Parameters
        ----------
        cond : int or str
            The condition label used to filter trials.
        merged : bool
            If True, returns a single flat list of all sizes across matching
            trials. If False, returns a list of lists, one per matching trial.

        Returns
        -------
        list
            Avalanche sizes for trials matching the specified condition,
            either merged or nested.
        """
        sizes = []
        for trial in self.trials:
            if trial.trial_label == cond:
                if merged:
                    sizes = sizes + trial.sizes
                else:
                    sizes.append(trial.sizes)
        return sizes

    def save(self, out_path: str, prefix):
        """
        Serializes and saves the TrialAvalanches object to a pickle file.

        Parameters
        ----------
        out_path : str
            Path to the directory where the file will be saved.
        prefix : str
            Prefix string used to name the output file. The file will be
            saved as `{prefix}_avalanche_obj.pkl`.
        """

        with open(os.path.join(out_path, f"{prefix}_avalanche_obj.pkl"), 'wb') as f:
            pickle.dump(self, f)


class AvalanchePipeline(BaseEstimator, TransformerMixin):
    """
    Core pipeline for detecting neuronal avalanches
    from multi-trial neural time series data.
    Thresholds the data using either a fixed value or a standard deviation
    based method, then identifies avalanches based on spatiotemporal
    activity patterns, filtering them by minimum size and duration. Scikit-learn compatible.

    Warning
    ----------
    Currently the Scikit-learn compatibility ("fit"/"transform") is implemented
    solely to allow "AvalanchePipeline" to be used as a preprocessing
    step upstream of "ATMPipeline" within a scikit-learn "Pipeline".
    It is NOT designed to function as a standalone scikit-learn
    transformer and may not behave as expected outside of that context.

    Parameters
    ----------
    z_scoring: bool
        If 'by_trial', z-scoring is applied independently to each trial.
        If 'by_channel', z-scoring is applied per channel across all trials
        and time points combined.
        If False, z-scoring is not applied. In that case, the input data should be ALREADY z-scored;
        the pipeline is supposed to work on z-scored data
    threshold_method : str
        Method used to threshold the data. Must be one of:
        - 'fixed': applies a fixed threshold value.
        - 'std': applies a threshold based on the signal's standard deviation (mean + threshold + std).
    threshold : float
        Threshold value used to binarize the data. Interpreted according
        to `threshold_method`.
    min_duration : int, optional
        Minimum duration in time bins for a valid avalanche.
        Defaults to 1.
    min_size : int, optional
        Minimum number of activations for a valid avalanche.
        Defaults to 1.
    bins : int, optional
        Number of time bins used for time subsampling. Defaults to 1.
    trial_label : int, optional
        If specified, only trials with this label are processed.
        Defaults to None.

    Raises
    ------
    ValueError
        If `threshold_method` is not one of the allowed methods
        ('fixed', 'std').
    """

    def __init__(
            self,
            z_scoring: Union[str, bool],
            threshold_method: str,
            threshold: float,
            min_duration: int = 1,
            min_size: int = 1,
            bins: int = 1,
            trial_label: int = None
            # z_score_by_trial: bool,
            # this is to select just a subset of trials based on their label (motor imagery, rest...)
    ):
        allowed_zscoring = {"by_channel", "by_trial", False}

        if z_scoring not in allowed_zscoring:
            raise ValueError(
                f"z_score must be one of {allowed_zscoring}, got '{z_scoring}'"
            )

        allowed_methods = {"fixed", "std"}

        if threshold_method not in allowed_methods:
            raise ValueError(
                f"method must be one of {allowed_methods}, got '{threshold_method}'"
            )

        self.z_scoring = z_scoring
        self.threshold_method = threshold_method
        self.threshold = threshold
        self.min_duration = min_duration
        self.min_size = min_size
        self.trial_label = trial_label
        self.bins = bins

    def __repr__(self):
        return (
            f"AvalanchePipeline(threshold={self.threshold}, "
            f"min_duration={self.min_duration}, "
            f"min_size={self.min_size})"
        )

    def _threshold_data_fixed(self, trial_data: np.ndarray):
        """
        Binarizes trial data using a fixed threshold.

        Sets to 1 all values whose absolute value exceeds `self.threshold`,
        and to 0 otherwise. Prints a warning if no activations are found.

        Parameters
        ----------
        trial_data : np.ndarray
            Neural time series data for a single trial of shape
            (n_channels, n_time_points).

        Returns
        -------
        np.ndarray
            Binarized data array of the same shape as `trial_data`.
        """
        binarized_data = np.where(np.abs(trial_data) > self.threshold, 1, 0)
        if not np.any(binarized_data):  # check if all elements are zero -> no signal above threshold gives a warning
            print(f'WARNING: No signal found above threshold {self.threshold}')

        return binarized_data

    def _threshold_data_std(self, trial_data: np.ndarray, mean, std):
        """Binarize trial data using a standard deviation-based threshold.

        Sets values to 1 when their absolute value exceeds the threshold
        (mean + threshold * std), and to 0 otherwise. Prints a warning
        if no activations are found above the threshold.

        Parameters
        ----------
        trial_data : np.ndarray
            Neural time series data for a single trial of shape
            (n_channels, n_time_points).
        mean : float
            Mean value of the data used for threshold calculation.
        std : float
            Standard deviation of the data used for threshold calculation.

        Returns
        -------
        np.ndarray
            Binarized data array of the same shape as `trial_data`,
            containing 1 for values above threshold and 0 otherwise.

        Warnings
        --------
        Prints a warning if no signal is found above the specified threshold.
        """
        binarized_data = np.where(
            np.abs(trial_data) > (mean + self.threshold * std), 1, 0
        )
        if not np.any(binarized_data):  # check if all elements are zero -> no signal above threshold gives a warning
            print(f'WARNING: No signal found above threshold {self.threshold}')

        return binarized_data

    def _compute_avalanches(self, bin_data: np.ndarray):
        """
        Detects and characterizes neuronal avalanches from binarized data.

        Identifies contiguous segments of activity in the binarized signal
        and extracts their spatiotemporal properties. Avalanches not meeting
        the minimum size and duration criteria are discarded.

        Parameters
        ----------
        bin_data : np.ndarray
            Binarized neural time series data of shape
            (n_channels, n_time_points).

        Returns
        -------
        matrices : list of np.ndarray
            Binarized spatiotemporal segments for each detected avalanche.
        sizes : list of int
            Total number of activations for each avalanche.
        durations : list of int
            Duration in time bins of each avalanche.
        patterns : list of np.ndarray
            Binarized spatial activation pattern for each avalanche.
        patterns_temporal : list of np.ndarray
            First activation time for each channel in each avalanche.
            Zero if the channel was never active.
        avalanche_bounds : list of tuple
            Start and end time bin indices for each avalanche.
        flexibility : int
            Number of distinct spatial activation patterns across all
            detected avalanches.
        avalanche_count : int
            Total number of detected avalanches.

        Notes
        -----
        If `self.bins` is not 1, the data is temporally subsampled by
        summing activations over non-overlapping windows of size `self.bins`
        before avalanche detection.
        """

        # check if there is at least one activation in the binarized signal
        if not np.any(bin_data):  # check if all elements are zero -> no signal above threshold gives a warning
            print(f"WARNING: No activations above threshold found in this trial")
            # warnings.warn(f"No activations above threshold found in this trial")

            matrices = []
            patterns = []
            patterns_temp = []
            sizes = []
            durations = []
            avalanche_bounds = []
            flexibility = 0
            avalanche_count = 0

            return matrices, sizes, durations, patterns, patterns_temp, avalanche_bounds, flexibility, avalanche_count

        # optional temporal subsampling
        if self.bins != 1:
            k = self.bins
            remainder = bin_data.shape[1] % k
            bin_data = np.column_stack(
                [np.sum(bin_data[:, i:i + k], axis=1) for i in range(0, bin_data.shape[1] - remainder, k)])

        # time points where there is at least one activation
        activations_bins = np.where(np.any(bin_data, axis=0))[0]
        # index difference between time points with activations
        diff = np.diff(activations_bins)
        # if the difference is not 1, the time points are not consecutive -> end of the avalanche
        split_points = np.where(diff != 1)[0]
        # avalanche starting points
        start_indices = [activations_bins[0]] + [activations_bins[i + 1] for i in split_points]
        # avalanche end points
        end_indices = [activations_bins[i] for i in split_points] + [activations_bins[-1]]

        matrices = []
        patterns = []
        patterns_temporal = []
        sizes = []
        durations = []
        avalanche_bounds = []

        # extract the avalanche segments and their characteristics
        for start, end in zip(start_indices, end_indices):
            segment_binary = bin_data[:, start:end + 1]
            duration = segment_binary.shape[1]

            # total number of activations for each region for an avalanche
            pattern_not_bin = np.sum(segment_binary, axis=1)
            pattern = np.where(pattern_not_bin > 0, 1, 0)  # binarization
            size = np.sum(pattern_not_bin)

            #  bool for each region, True if there is at least one activation
            is_active = segment_binary.any(axis=1)
            # return the time when the region activated the first time if it was ever active, otherwise zero
            pattern_temporal = np.where(is_active, np.argmax(segment_binary, axis=1) + 1, 0)

            # check if the avalanche satisfies the duration and size criteria
            if duration >= self.min_duration and size >= self.min_size:
                matrices.append(segment_binary)
                patterns.append(pattern)
                patterns_temporal.append(pattern_temporal)
                avalanche_bounds.append((start, end))
                durations.append(duration)
                sizes.append(size)

        # check if at least one avalanche is found
        if len(matrices) == 0:  # no avalanche was above the min duration and size
            print(f"No avalanches satisfied the minimum duration and size constraint for this trial")

            flexibility = 0
            avalanche_count = 0

            return matrices, sizes, durations, patterns, patterns_temporal, avalanche_bounds, flexibility, avalanche_count

        flexibility = len(np.unique(patterns, axis=0))
        avalanche_count = len(matrices)

        return matrices, sizes, durations, patterns, patterns_temporal, avalanche_bounds, flexibility, avalanche_count

    #def run(self, dataset: Dataset) -> TrialAvalanches:
    #def run(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, np.ndarray], info: dict = None) -> TrialAvalanches:
    def process(self, data: Union[mne.io.BaseRaw, mne.BaseEpochs, dict]) -> TrialAvalanches:
        """Run the avalanche detection pipeline on neural recording data.

        Iterates through all trials in the dataset (or a subset filtered by
        `self.trial_label`), applies z-score normalization, binarizes the data
        using the specified threshold method, detects avalanches, and returns
        the results as a TrialAvalanches object.

        The pipeline performs the following steps:
        1. Constructs a Dataset object from the input data
        2. Z-scores the data (optionally by trial)
        3. Filters trials by `self.trial_label` if specified
        4. Binarizes each trial using either 'fixed' or 'std' thresholding
        5. Detects avalanches and computes metrics (size, duration, patterns, flexibility)

        Parameters
        ----------
        data : mne.io.BaseRaw, mne.BaseEpochs, or dict
            The input neural recording data. Options are:

            - mne.io.BaseRaw : Raw continuous recording object
            - mne.BaseEpochs : Epoched recording object
            - dict : Custom data dictionary with required keys:
                - 'data_array' : np.ndarray of shape (n_trials, n_channels, n_timepoints)
                - 'samp_freq' : Sampling frequency in Hz
                - 'fmin' : Minimum frequency
                - 'fmax' : Maximum frequency
                - 'channel_names' : List of channel names
                - 'trial_labels' : List of trial condition labels
                - 't_max' : Maximum time points to upload

        Returns
        -------
        TrialAvalanches
            A TrialAvalanches object containing one AvalancheObj per
            processed trial with avalanche metrics (size, duration, patterns,
            flexibility, bounds).

        Raises
        ------
        ValueError
            If `self.trial_label` is specified but no trials with that
            label are found in the dataset.
        TypeError
            If data is not a recognized format (BaseRaw, BaseEpochs, or dict).
        ValueError
            If an invalid threshold method is specified (must be 'fixed' or 'std').

        Notes
        -----
        Status messages are printed during execution (dataset creation,
        z-scoring, binarization, avalanche detection).
        """

        dataset = _build_dataset(data)
        print("Dataset created")

        #dataset.zscore_data(by_trial=self.z_score_by_trial)

        if self.z_scoring == 'by_trial':
            dataset.zscore_data(by_trial=True)
        elif self.z_scoring == 'by_channel':
            dataset.zscore_data(by_trial=False)
        else:
            print('Warning: the data is NOT being z-scored.')
            dataset.data_zscored = dataset.data

        #### original code starts from here

        list_results = []

        # check the input trial label is present in the input data
        if self.trial_label is not None:
            trial_indexes = [idx for idx, lab in enumerate(dataset.trial_labels) if lab == self.trial_label]
            if len(trial_indexes) == 0:
                raise ValueError(
                    f"No trials found with label {self.trial_label}"
                )
            else:
                print(f"Found {len(trial_indexes)} trials with label {self.trial_label}")

        else:
            trial_indexes = dataset.trial_ids

        # iterate through the trials to compute the avalanches
        for trial_idx in trial_indexes:
            print(f"Computing trial {trial_idx}...")
            #trial_data = dataset.data[trial_idx]
            trial_data = dataset.data_zscored[trial_idx]
            # binarize data
            print("Binarizing data...")
            if self.threshold_method == 'fixed':
                bin_data = self._threshold_data_fixed(trial_data)
            elif self.threshold_method == 'std':
                mean, std = dataset._get_mean_and_std()
                bin_data = self._threshold_data_std(trial_data, mean, std)
            else:
                raise ValueError(
                    f"No valid thresholding method was given'"
                )

            # compute avalanches
            print("Computing avalanches...")
            matrices, sizes, durations, patterns, patterns_temporal, avalanche_bounds, flexibility, avalanche_count = self._compute_avalanches(
                bin_data)

            # store results for each trial
            trial_result = AvalancheObj(#trial_id=trial_idx,
                                        trial_id=trial_idx,
                                        avalanches_mat=matrices,
                                        time_bins=self.bins,
                                        sizes=sizes,
                                        durations=durations,
                                        patterns=patterns,
                                        patterns_temporal=patterns_temporal,
                                        bounds=avalanche_bounds,
                                        flexibility=flexibility,
                                        avalanche_count=avalanche_count)
            # if self.trial_label is not None:
            # trial_result.trial_label = self.trial_label
            # TODO qui succede che se ci sono i trials non prende i labels, se é rest si rompe tutto, risolvi
            if dataset.trial_labels is not None:
                trial_result.trial_label = dataset.trial_labels[trial_idx]

            list_results.append(trial_result)

            print('Done.')

        # store all trials results
        results = TrialAvalanches(list_results)
        return results

    # methods for sklearn Pipeline compatibility
    def fit(self, X, y=None):
        """
        Placeholder fit method for scikit-learn Pipeline compatibility.
        This method does not perform any computation and simply returns self.
        """
        return self

    def transform(self, X: np.ndarray):
        """
        Runs the avalanche detection pipeline on a raw NumPy array.

        Iterates through all trials in the input array, binarizes the data,
        detects avalanches, and returns the results as a TrialAvalanches
        object. Intended for use within a scikit-learn Pipeline.

        Warning
        ----------
        Currently the Scikit-learn compatibility ("fit"/"transform") is implemented
        solely to allow "AvalanchePipeline" to be used as a preprocessing
        step upstream of "ATMPipeline" within a scikit-learn "Pipeline".
        It is NOT designed to function as a standalone scikit-learn
        transformer and may not behave as expected outside of that context.

        Parameters
        ----------
        X : np.ndarray
            Neural time series data of shape
            (n_trials, n_channels, n_time_points).

        Returns
        -------
        TrialAvalanches
            A TrialAvalanches object containing one AvalancheObj per
            processed trial.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be a numpy array, got {type(X)} instead")

        dataset = Dataset(data=X, samp_freq=1, fmax=1, fmin=1) # the other arguments are placeholders - not needed here
        dataset.zscore_data(by_trial=False)
        mean_data, std_data = dataset._get_mean_and_std()

        list_results = []
        #print('shape: ', np.shape(X))
        trial_indexes = range(0, np.shape(X)[0])

        # iterate through the trials to compute the avalanches
        for trial_idx in trial_indexes:
            print(f"Computing trial {trial_idx}...")
            trial_data_orig = X[trial_idx]
            # z-scoring
            trial_data = zscore(trial_data_orig)
            # time series binarization with the chosen method
            print("Binarizing data...")
            if self.threshold_method == 'fixed':
                bin_data = self._threshold_data_fixed(trial_data)
            elif self.threshold_method == 'std':
                bin_data = self._threshold_data_std(trial_data, mean_data, std_data)
            else:
                raise ValueError(
                    f"No valid thresholding method was given'"
                )

            # compute avalanches in the standard way
            print("Computing avalanches...")
            matrices, sizes, durations, patterns, patterns_temporal, avalanche_bounds, flexibility, avalanche_count = self._compute_avalanches(
                bin_data)

            # create single trial results object
            trial_result = AvalancheObj(trial_id=trial_idx,
                                        avalanches_mat=matrices,
                                        time_bins=self.bins,
                                        sizes=sizes,
                                        durations=durations,
                                        patterns=patterns,
                                        patterns_temporal=patterns_temporal,
                                        bounds=avalanche_bounds,
                                        flexibility=flexibility,
                                        avalanche_count=avalanche_count)

            list_results.append(trial_result)
            print('Done.')

        # store results of all trials
        results = TrialAvalanches(list_results)
        return results

##### plot avalanches distributions with binning

ALLOWED_PLOTS            = {"size", "duration"}
ALLOWED_THRESHOLD_METHODS = {"fixed", "std"}

#: Axis labels and figure titles, keyed by the `plot` argument.
_PLOT_LABELS = {
    "size":     ("Avalanche Size",     "Avalanche Size Distribution"),
    "duration": ("Avalanche Duration", "Avalanche Duration Distribution"),
}

def _validate_arguments(plot: str, threshold_method: str, linear: int,
                        time_bins: Sequence[int], hist_bins: int) -> None:
    """
    Validates every user-supplied argument before any computation starts.

    Raises
    ------
    ValueError
        If any argument is outside its allowed set or range.
    """
    if plot not in ALLOWED_PLOTS:
        raise ValueError(f"plot must be one of {ALLOWED_PLOTS}, got {plot!r}")

    if threshold_method not in ALLOWED_THRESHOLD_METHODS:
        raise ValueError(
            f"threshold_method must be one of {ALLOWED_THRESHOLD_METHODS}, "
            f"got {threshold_method!r}"
        )

    if linear not in (0, 1):
        raise ValueError(f"linear must be either 0 or 1, got {linear!r}")

    if not len(time_bins):
        raise ValueError("time_bins must contain at least one bin size")

    if any(k < 1 for k in time_bins):
        raise ValueError(f"time_bins entries must be >= 1, got {list(time_bins)}")

    if hist_bins < 1:
        raise ValueError(f"hist_bins must be >= 1, got {hist_bins}")

def _flatten_trials(data: np.ndarray) -> np.ndarray:
    """
    Concatenates all trials along the time axis.
    (n_trials, n_channels, n_time) -> (n_channels, n_trials * n_time)
    """
    # Move channels to axis 0, then merge the trial and time axes.
    channels_first = np.transpose(data, (1, 0, 2))
    return channels_first.reshape(channels_first.shape[0], -1)

def _binarize(data_flat: np.ndarray, threshold_method: str,
              threshold: float, mean: float, std: float) -> np.ndarray:
    """
    Marks every sample whose absolute value exceeds the threshold as 1.

    Parameters
    ----------
    data_flat : np.ndarray
        Z-scored data of shape (n_channels, n_time).
    threshold_method : {'fixed', 'std'}
        'fixed' uses `threshold` directly; 'std' uses
        `mean + threshold * std`.
    threshold : float
        Threshold value or standard-deviation multiplier.
    mean : float
        Mean value of z-scored data across all trials

    Returns
    -------
    np.ndarray
        Binary array with the same shape as `data_flat`.

    Raises
    ------
    ValueError
        If no sample exceeds the threshold, since there is nothing to analyse.
    """
    cutoff = threshold if threshold_method == "fixed" else mean + threshold * std
    binarized = np.where(np.abs(data_flat) > cutoff, 1, 0)

    if not np.any(binarized):
        raise ValueError(
            f"No activations above threshold {threshold} "
            f"(method={threshold_method!r}). Try lowering the threshold."
        )
    return binarized

def _subsample_time(binarized: np.ndarray, k: int) -> np.ndarray:
    """
    Merges every k consecutive time bins by summing their activations.

    Parameters
    ----------
    binarized : np.ndarray
        Binary array of shape (n_channels, n_time).
    k : int
        Bin size. k == 1 returns the input unchanged.

    Returns
    -------
    np.ndarray
        Array of shape (n_channels, n_time // k).
    """
    if k == 1:
        return binarized

    n_channels, n_time = binarized.shape
    n_bins = n_time // k

    # Reshape into (channels, n_bins, k) and sum over the last axis - checked ok
    trimmed = binarized[:, :n_bins * k]
    return trimmed.reshape(n_channels, n_bins, k).sum(axis=2)

def _find_avalanches(binarized: np.ndarray):
    """Find contiguous runs of neural activity across time.

    An avalanche is defined as a contiguous period where at least one
    channel shows activity. It starts when the first channel becomes active
    and ends at the last consecutive time bin with any activity.

    Parameters
    ----------
    binarized : np.ndarray
        Binary array of shape (n_channels, n_time_bins) where non-zero
        values indicate active channels.

    Returns
    -------
    start_indices : list of int
        Start time bin indices (inclusive) for each avalanche.
    end_indices : list of int
        End time bin indices (inclusive) for each avalanche.

    Notes
    -----
    Both lists are empty if there is no activity in the data.
    The indices correspond to time bins where activity occurs.
    """
    # time points where there is at least one activation
    activations_bins = np.where(np.any(binarized, axis=0))[0]
    # index difference between time points with activations
    diff = np.diff(activations_bins)
    # if the difference is not 1, the time points are not consecutive -> end of the avalanche
    split_points = np.where(diff != 1)[0]
    # avalanche start indices
    start_indices = [activations_bins[0]] + [activations_bins[i + 1] for i in split_points]
    # avalanche end indices
    end_indices = [activations_bins[i] for i in split_points] + [activations_bins[-1]]

    return start_indices, end_indices

def _avalanche_stats(binarized: np.ndarray, k: int) -> Dict[str, List]:
    """Compute size and duration statistics for all avalanches.

    Extracts contiguous periods of neural activity and quantifies each
    avalanche by the total number of activations (size) and its temporal
    extent (duration).

    Parameters
    ----------
    binarized : np.ndarray
        Binary array of shape (n_channels, n_time_bins) where non-zero
        values indicate active channels.
    k : int
        Time bin size used for subsampling. Durations are scaled by k
        to express results in original time samples regardless of k.

    Returns
    -------
    dict
        Dictionary with two keys:

        - 'size' : list of int
            Total number of activations per avalanche (summed across channels).
        - 'duration' : list of int
            Length (in time bins) of each avalanche, scaled by k to match
            the original sampling rate.

    Notes
    -----
    Durations are always scaled by k to ensure consistency between
    subsampled and original time units. This guarantees that global
    and per-condition analyses use the same time units.
    """
    subsampled = _subsample_time(binarized, k)
    start_indices, end_indices = _find_avalanches(subsampled)

    sizes = []
    durations = []

    # extract the avalanche segments and their characteristics
    for start, end in zip(start_indices, end_indices):
        segment_binary = subsampled[:, start:end + 1]
        duration = segment_binary.shape[1]

        # total number of activations for each region for an avalanche
        pattern_not_bin = np.sum(segment_binary, axis=1)
        # pattern = np.where(pattern_not_bin > 0, 1, 0)  # binarization
        size = np.sum(pattern_not_bin)

        durations.append(duration)
        sizes.append(size)

    if k!=1:
        np.multiply(durations, k)

    return {"size": sizes, "duration": durations}

def _probability_density(values: np.ndarray, hist_bins: int,
                         linear: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized probability density from avalanche values.

    Calculates the probability density function (PDF) of avalanche sizes
    or durations using either logarithmic or linear binning.

    Parameters
    ----------
    values : np.ndarray
        Array of avalanche sizes or durations to analyze.
    hist_bins : int
        Number of histogram bins. Higher values provide finer resolution.
    linear : {0, 1}
        Binning type:

        - 0 : logarithmic bin edges (useful for power-law distributions)
        - 1 : linear bin edges (uniform spacing)

    Returns
    -------
    x : np.ndarray
        Bin centers (avalanche values).
    y : np.ndarray
        Normalized probability density at each bin.

    Notes
    -----
    Only bins with strictly positive, finite density are returned.
    This filtering ensures the data is suitable for visualization on
    log-log axes without NaNs or infinities.

    Degenerate case: If all input values are identical, returns a single
    bin with probability density of 1.0.
    """
    values = np.asarray(values, dtype=float)
    n      = values.size
    xmin, xmax = values.min(), values.max()

    # Degenerate case: all values identical, so the whole mass sits in one bin.
    if xmin == xmax:
        return np.array([xmin]), np.array([1.0])

    if linear == 0:

        bin_centers = np.zeros(hist_bins)
        bin_widths = np.zeros(hist_bins)

        for i in range(hist_bins):
            bin_centers[i] = xmin * np.exp((i + 0.5) * np.log(xmax / xmin) / hist_bins)
            bin_widths[i] = xmin * (np.exp((i + 1) * np.log(xmax / xmin) / hist_bins)
                                    - np.exp(i * np.log(xmax / xmin) / hist_bins))

        # Count occurrences in each bin
        bin_counts = np.zeros(hist_bins)
        for value in values:
            bin_idx = int(np.ceil(hist_bins * np.log(value / xmin) / np.log(xmax / xmin)))
            if 1 <= bin_idx <= hist_bins:
                bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

        # Normalized probability density
        prob_distribution = bin_counts / (bin_widths * n)

    # ── Linear Bins ────────────────────────────────────────────────────────────
    else:

        bin_counts = np.zeros(hist_bins)
        bin_centers = np.zeros(hist_bins)

        # Count occurrences in each bin
        for value in values:
            bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
            if 1 <= bin_idx <= hist_bins:
                bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

        for i in range(hist_bins):
            bin_centers[i] = xmin + (i + 0.5) * (xmax - xmin) / hist_bins

        # Normalized probability density
        prob_distribution = bin_counts * hist_bins / (n * (xmax - xmin))

    mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

    x = bin_centers[mask]
    y = prob_distribution[mask]

    return x, y

def _plot_distribution(ax, binarized: np.ndarray, plot: str,
                       time_bins: Sequence[int], hist_bins: int,
                       linear: int) -> Dict[int, np.ndarray]:
    """
    Draws one curve per time bin size onto a single axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    binarized : np.ndarray
        Binary data of shape (n_channels, n_time).
    plot : {'size', 'duration'}
        Which quantity to plot.
    time_bins : sequence of int
        Bin sizes; one curve is drawn per entry.
    hist_bins : int
        Number of histogram bins.
    linear : {0, 1}
        Binning strategy.

    Returns
    -------
    dict
        {time_bin: raw values} - the data behind each plotted curve.
    """
    colors  = plt.cm.tab10(np.linspace(0, 1, len(time_bins)))
    results = {}

    for color, k in zip(colors, time_bins):
        values = _avalanche_stats(binarized, k)[plot]
        results[k] = values

        # An empty result means no avalanche survived at this bin size,
        # e.g. because k exceeds the recording length. Skip rather than crash.
        if len(values) == 0:
            continue

        x, y = _probability_density(values, hist_bins, linear)
        if x.size == 0:
            continue

        # scatter for the markers, plot for the connecting line; zorder keeps
        # the markers above the line.
        ax.scatter(x, y, color=color, edgecolors="black", s=60, zorder=3,
                   label=f"time bins = {k}")
        ax.plot(x, y, color=color, linewidth=1.5, zorder=2)

    xlabel, _ = _PLOT_LABELS[plot]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Probability Density", fontsize=13)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    if ax.get_legend_handles_labels()[0]:      # only if something was drawn
        ax.legend(fontsize=11)

    return results

def plot_avalanche_distribution(
        input_data: Union[mne.io.BaseRaw, mne.BaseEpochs, dict],
        plot: str,
        threshold_method: str,
        threshold: float,
        time_bins: List[int],
        hist_bins: int,
        linear: int,
) -> Tuple[plt.Figure, Optional[plt.Figure], dict]:
    """
    Plots the probability density of avalanche sizes or durations on log-log axes.

    The data is z-scored, binarized with the chosen thresholding method, and
    scanned for avalanches. One curve is drawn per entry in `time_bins`. If the
    dataset contains more than one condition, a second figure with one subplot
    per condition is produced as well.

    Parameters
    ----------
    input_data : mne.io.BaseRaw, mne.BaseEpochs or dict
        Input data. A dict must contain 'data_array', 'samp_freq', 'fmin',
        'fmax', 'channel_names', 'trial_labels' and 't_max'.
    plot : {'size', 'duration'}
        Quantity to plot.
    threshold_method : {'fixed', 'std'}
        'fixed' applies `threshold` directly to the z-scored data;
        'std' applies `threshold * std` of the z-scored data.
    threshold : float
        Threshold value or standard-deviation multiplier.
    time_bins : list of int
        Time bin sizes for temporal subsampling; one curve per entry.
    hist_bins : int
        Number of histogram bins.
    linear : {0, 1}
        0 for logarithmic binning, 1 for linear binning.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Global distribution across all trials and conditions.
    fig_cond : matplotlib.figure.Figure or None
        One subplot per condition, or None when the dataset has a single
        condition.
    results : dict
        {'global': {time_bin: values},
         'by_condition': {condition: {time_bin: values}}}
        'by_condition' is absent when there is only one condition.

    Raises
    ------
    TypeError
        If `input_data` is of an unsupported type.
    ValueError
        If any argument is invalid, or if no sample exceeds the threshold.

    Notes
    -----
    The return arity is now FIXED at three values. The original returned two
    values for a single condition and three otherwise, so callers could not
    unpack it safely. `fig_cond` is None in the single-condition case.
    """
    # validate arguments, then build the dataset

    _validate_arguments(plot, threshold_method, linear, time_bins, hist_bins)
    dataset = _build_dataset(input_data)

    # z-score
    # by_trial=False normalises each channel over all trials at once, keeping
    # relative amplitude differences between trials meaningful.
    dataset.zscore_data(by_trial=False)

    zscored = dataset.data_zscored

    # global figure
    mean, std = dataset._get_mean_and_std()
    global_binarized = _binarize(_flatten_trials(zscored),
                                 threshold_method, threshold, mean, std)

    fig, ax = plt.subplots(figsize=(8, 6))
    global_results = _plot_distribution(ax, global_binarized, plot,
                                        time_bins, hist_bins, linear)

    _, title = _PLOT_LABELS[plot]
    ax.set_title(title, fontsize=15)
    fig.tight_layout()

    results = {"global": global_results}

    # per-condition figure
    labels = dataset.trial_labels
    if labels is None or len(set(labels)) <= 1:
        return fig, None, results

    # sorted() makes the subplot order reproducible
    conditions = sorted(set(labels), key=str)

    # squeeze=False guarantees a 2-D axes array even for a single column,
    # so axs[0] is always iterable.
    fig_cond, axs = plt.subplots(1, len(conditions),
                                 figsize=(6 * len(conditions), 5),
                                 squeeze=False)

    by_condition = {}
    for cond, ax_cond in zip(conditions, axs[0]):
        print(f"Computing condition {cond}...")

        mask = np.array([lab == cond for lab in labels])
        cond_data = zscored[mask]

        cond_binarized = _binarize(_flatten_trials(cond_data),
                                   threshold_method, threshold, mean, std)

        by_condition[cond] = _plot_distribution(ax_cond, cond_binarized, plot,
                                                time_bins, hist_bins, linear)
        ax_cond.set_title(f"Condition {cond}", fontsize=13)

    fig_cond.suptitle(title, fontsize=15)
    fig_cond.tight_layout()

    results["by_condition"] = by_condition

    return fig, fig_cond, results


def _plot_avalanche_distribution_deprecated(input_data: Union[mne.io.BaseRaw, mne.BaseEpochs, dict], plot: str, threshold_method: str, threshold: int,
                                time_bins: List[int], hist_bins: int, linear: int):
    """
    Plots the probability density distribution of avalanche sizes or durations
    on a log-log scale, for multiple time bin sizes and conditions.

    The data is first binarized using the specified thresholding method, then
    avalanches are detected and their sizes or durations are computed. The
    distribution is plotted for each time bin size specified. If the dataset
    contains multiple conditions, a separate subplot is generated for each
    condition in addition to the global plot.

    Parameters
    ----------
    dataset : Dataset
        A Dataset object containing the neural time series data and associated
        metadata.
    plot : str
        The quantity to plot. Must be one of:
        - 'size': plots the avalanche size distribution.
        - 'duration': plots the avalanche duration distribution.
    threshold_method : str
        Method used to binarize the data. Must be one of:
        - 'fixed': applies a fixed threshold value.
        - 'std': applies a threshold based on the signal's standard deviation.
    threshold : int
        Threshold value used to binarize the data. Interpreted according
        to `threshold_method`.
    time_bins : list of int
        List of time bin sizes to use for avalanche detection. A separate
        curve is plotted for each bin size.
    hist_bins : int
        Number of bins used to compute the probability density histogram.
    linear : int
        Binning strategy for the histogram. Must be one of:
        - 0: logarithmic binning.
        - 1: linear binning.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the global avalanche distribution plot across all
        trials and conditions.
    fig_cond : matplotlib.figure.Figure
        Figure containing one subplot per condition, each showing the
        avalanche distribution for that condition. Only returned if the
        dataset contains more than one condition.
    dict_results : dict
        Dictionary containing the raw avalanche size or duration values used
        for plotting. If a single condition is present, keys are time bin
        sizes. If multiple conditions are present, keys are condition labels,
        each mapping to a nested dictionary keyed by time bin size.

    Raises
    ------
    ValueError
        If `linear` is not 0 or 1, if `threshold_method` is not one of the
        allowed methods, if `plot` is not one of the allowed options, or if
        no activations are found above the threshold.

    Notes
    -----
    When multiple conditions are present, the data is first flattened across
    all trials and conditions for the global plot, then split by condition
    for the per-condition plot. Avalanche durations are scaled by the time
    bin size `k` to reflect the original time scale.
    """

    if isinstance(input_data, mne.io.BaseRaw):
        dataset = create_dataset_from_mne(input_data)

    elif isinstance(input_data, mne.BaseEpochs):
        dataset = create_dataset_from_mne(input_data)

    elif isinstance(input_data, dict):
        dataset = Dataset(
            data=input_data["data_array"],
            samp_freq=input_data["samp_freq"],
            fmin=input_data["fmin"],
            fmax=input_data["fmax"],
            channel_names=input_data["channel_names"],
            # trial_ids=info["trial_ids"],
            trial_labels=input_data["trial_labels"],
            t_max=input_data["t_max"],
        )

    #elif isinstance(input_data, np.ndarray) and info is None:
    #    raise TypeError("When a np.ndarray is provided, an info dictionary is needed")
    else:
        raise TypeError("Input must be Raw or Epochs object or dict")

    #TODO check how and IF this should be z-scored
    dataset.zscore_data(by_trial=False)

    # check logarithmic or linear binning
    if linear not in [0, 1]:
        raise ValueError("Linear must be either 0 or 1")

    # check the thresholding method for time-series binarization
    allowed_methods = {"fixed", "std"}

    if threshold_method not in allowed_methods:
        raise ValueError(
            f"method must be one of {allowed_methods}, got '{threshold_method}'"
        )

    # check what kind of data to plot
    allowed_plots = {"size", "duration"}

    if plot not in allowed_plots:
        raise ValueError(
            f"plot must be one of {allowed_plots}, got '{plot}'"
        )

    # extract the z-scored time series from the Dataset object
    data = dataset.data_zscored

    # flatten the data by concatenating all trials
    temp = np.transpose(data, (1, 0, 2))
    data_flat = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))

    print("Binarizing data...")
    if threshold_method == 'fixed':
        bin_data_orig = np.where(np.abs(data_flat) > threshold, 1, 0)
    elif threshold_method == 'std':
        mean, std = dataset._get_mean_and_std()
        bin_data_orig = np.where(np.abs(data_flat) > (mean + threshold * std), 1, 0)

    # check if there is at least one activation in the binarized signal
    if not np.any(bin_data_orig):  # check if all elements are zero -> no signal above threshold gives a warning
        raise ValueError("No activations above threshold found")

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(time_bins)))

    dict_results = {}

    for color, k in zip(colors, time_bins):
        print(f"Computing avalanches for {k} bins...")
        if k != 1:
            # if there is time subsampling, collapse the data of k time bins by summing the activations
            remainder = bin_data_orig.shape[1] % k
            bin_data = np.column_stack(
                [np.sum(bin_data_orig[:, i:i + k], axis=1) for i in range(0, bin_data_orig.shape[1] - remainder, k)])
        else:
            bin_data = bin_data_orig.copy()

        # time points where there is at least one activation
        activations_bins = np.where(np.any(bin_data, axis=0))[0]
        # index difference between time points with activations
        diff = np.diff(activations_bins)
        # if the difference is not 1, the time points are not consecutive -> end of the avalanche
        split_points = np.where(diff != 1)[0]
        # avalanche start indices
        start_indices = [activations_bins[0]] + [activations_bins[i + 1] for i in split_points]
        # avalanche end indices
        end_indices = [activations_bins[i] for i in split_points] + [activations_bins[-1]]

        sizes = []
        durations = []

        # extract the avalanche segments and their characteristics
        for start, end in zip(start_indices, end_indices):
            segment_binary = bin_data[:, start:end + 1]
            duration = segment_binary.shape[1]

            # total number of activations for each region for an avalanche
            pattern_not_bin = np.sum(segment_binary, axis=1)
            # pattern = np.where(pattern_not_bin > 0, 1, 0)  # binarization
            size = np.sum(pattern_not_bin)

            durations.append(duration)
            sizes.append(size)

        # TODO moltiplica durate per k
        # if there was a time subsampling, adjust the durations accordingly
        durations = np.multiply(durations, k)

        if plot == "size":
            data_to_plot = sizes
        elif plot == "duration":
            data_to_plot = durations

        dict_results[k] = data_to_plot

        Ndat = len(data_to_plot)
        xmin = np.min(data_to_plot)
        xmax = np.max(data_to_plot)

        # ── Logarithmic Bins ───────────────────────────────────────────────────────
        if linear == 0:

            bin_centers = np.zeros(hist_bins)
            bin_widths = np.zeros(hist_bins)

            for i in range(hist_bins):
                bin_centers[i] = xmin * np.exp((i + 0.5) * np.log(xmax / xmin) / hist_bins)
                bin_widths[i] = xmin * (np.exp((i + 1) * np.log(xmax / xmin) / hist_bins)
                                        - np.exp(i * np.log(xmax / xmin) / hist_bins))

            # Count occurrences in each bin
            bin_counts = np.zeros(hist_bins)
            for value in data_to_plot:
                bin_idx = int(np.ceil(hist_bins * np.log(value / xmin) / np.log(xmax / xmin)))
                if 1 <= bin_idx <= hist_bins:
                    bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

            # Normalized probability density
            prob_distribution = bin_counts / (bin_widths * Ndat)

        # ── Linear Bins ────────────────────────────────────────────────────────────
        elif linear == 1:

            bin_counts = np.zeros(hist_bins)
            bin_centers = np.zeros(hist_bins)

            # Count occurrences in each bin
            for value in data_to_plot:
                bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
                if 1 <= bin_idx <= hist_bins:
                    bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

            for i in range(hist_bins):
                bin_centers[i] = xmin + (i + 0.5) * (xmax - xmin) / hist_bins

            # Normalized probability density
            prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

        mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

        x = bin_centers[mask]
        y = prob_distribution[mask]

        # Plot

        ax.scatter(x, y,
                   color=color,
                   edgecolors="black",
                   s=60,
                   zorder=3,
                   label=f"time bins = {k}")

        ax.plot(x, y,
                color=color,
                linewidth=1.5,
                zorder=2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        if plot == "size":
            ax.set_xlabel("Avalanche Size", fontsize=13)
            ax.set_title("Avalanche Size Distribution", fontsize=15)
        else:
            ax.set_xlabel("Avalanche Duration", fontsize=13)
            ax.set_title("Avalanche Duration Distribution", fontsize=15)

        ax.set_ylabel("Probability Density", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()

    if dataset.trial_labels is None or len(set(dataset.trial_labels)) == 1:  # if it's rest or just one trial condition
        return fig, dict_results

    else:  # if I have more than one trial label repeat the computation for each condition
        conditions = set(dataset.trial_labels)
        # results_cond = []
        dict_results = {}
        fig_cond, axs = plt.subplots(1, len(conditions))
        for cond, ax in zip(conditions, axs):
            dict_cond = {}
            print(f"Computing condition {cond}...")
            count = sum(1 for value in dataset.trial_labels if value == cond)
            data_condition = np.zeros((count, dataset.n_channels, dataset.n_time_points))
            for i, j in zip(range(len(dataset.trial_ids)), range(count)):
                if dataset.trial_labels[i] == cond:
                    data_condition[j, :, :] = dataset.get_trial(i)

            data = data_condition

            temp = np.transpose(data, (1, 0, 2))
            data_flat = np.reshape(temp, (np.shape(temp)[0], np.shape(temp)[1] * np.shape(temp)[2]))

            print("Binarizing data...")
            if threshold_method == 'fixed':
                bin_data_orig = np.where(np.abs(data_flat) > threshold, 1, 0)
            elif threshold_method == 'std':
                std = data_flat.std()
                bin_data_orig = np.where(np.abs(data_flat) > (threshold * std), 1, 0)

            # check if there is at least one activation in the binarized signal
            if not np.any(bin_data_orig):  # check if all elements are zero -> no signal above threshold gives a warning
                raise ValueError("No activations above threshold found")

            colors = plt.cm.tab10(np.linspace(0, 1, len(time_bins)))

            for color, k in zip(colors, time_bins):
                # for ax, k in zip(axes, bins):
                print(f"Computing avalanches for {k} bins...")
                if k != 1:
                    remainder = bin_data_orig.shape[1] % k
                    bin_data = np.column_stack(
                        [np.sum(bin_data_orig[:, i:i + k], axis=1) for i in
                         range(0, bin_data_orig.shape[1] - remainder, k)])
                else:
                    bin_data = bin_data_orig.copy()

                activations_bins = np.where(np.any(bin_data, axis=0))[
                    0]  # indici dei time points dove c'é stata almeno un'attivazione

                diff = np.diff(activations_bins)  # guardo la differenza tra indici di dove ho avuto attivazioni
                split_points = np.where(diff != 1)[
                    0]  # punti dove gli indici non sono consecutivi -> finisce la valanga
                start_indices = [activations_bins[0]] + [activations_bins[i + 1] for i in
                                                         split_points]  # punti di inizio delle valanghe
                end_indices = [activations_bins[i] for i in split_points] + [
                    activations_bins[-1]]  # punti di fine delle valanghe

                sizes = []
                durations = []

                # extract the avalanche segments and their characteristics
                for start, end in zip(start_indices, end_indices):
                    segment_binary = bin_data[:, start:end + 1]
                    # segment_binary = bin_data[:, start:end]
                    duration = segment_binary.shape[1]

                    pattern_not_bin = np.sum(segment_binary,
                                             axis=1)  # total number of activations for each region for an avalanche
                    # pattern = np.where(pattern_not_bin > 0, 1, 0)  # binarization
                    size = np.sum(pattern_not_bin)

                    durations.append(duration)
                    sizes.append(size)

                if plot == "size":
                    data_to_plot = sizes
                elif plot == "duration":
                    data_to_plot = durations

                dict_cond[k] = data_to_plot

                # dict_results[cond] = data_to_plot

                Ndat = len(data_to_plot)
                xmin = np.min(data_to_plot)
                xmax = np.max(data_to_plot)

                # ── Logarithmic Bins ───────────────────────────────────────────────────────
                if linear == 0:

                    bin_centers = np.zeros(hist_bins)
                    bin_widths = np.zeros(hist_bins)

                    for i in range(hist_bins):
                        bin_centers[i] = xmin * np.exp((i + 0.5) * np.log(xmax / xmin) / hist_bins)
                        bin_widths[i] = xmin * (np.exp((i + 1) * np.log(xmax / xmin) / hist_bins)
                                                - np.exp(i * np.log(xmax / xmin) / hist_bins))

                    # Count occurrences in each bin
                    bin_counts = np.zeros(hist_bins)
                    for value in data_to_plot:
                        bin_idx = int(np.ceil(hist_bins * np.log(value / xmin) / np.log(xmax / xmin)))
                        if 1 <= bin_idx <= hist_bins:
                            bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

                    # Normalized probability density
                    prob_distribution = bin_counts / (bin_widths * Ndat)

                # ── Linear Bins ────────────────────────────────────────────────────────────
                elif linear == 1:

                    bin_counts = np.zeros(hist_bins)
                    bin_centers = np.zeros(hist_bins)

                    # Count occurrences in each bin
                    for value in data_to_plot:
                        bin_idx = int(np.ceil(hist_bins * (value - xmin) / (xmax - xmin)))
                        if 1 <= bin_idx <= hist_bins:
                            bin_counts[bin_idx - 1] += 1  # Convert 1-based to 0-based index

                    for i in range(hist_bins):
                        bin_centers[i] = xmin + (i + 0.5) * (xmax - xmin) / hist_bins

                    # Normalized probability density
                    prob_distribution = bin_counts * hist_bins / (Ndat * (xmax - xmin))

                mask = (prob_distribution > 0) & np.isfinite(prob_distribution)

                x = bin_centers[mask]
                y = prob_distribution[mask]

                # Plot

                ax.scatter(x, y,
                           color=color,
                           edgecolors="black",
                           s=60,
                           zorder=3,
                           label=f"time bins = {k}")

                ax.plot(x, y,
                        color=color,
                        linewidth=1.5,
                        zorder=2)

                ax.set_xscale("log")
                ax.set_yscale("log")
                if plot == "size":
                    ax.set_xlabel("Avalanche Size", fontsize=13)
                    ax.set_title(f"Condition {cond}", fontsize=13)
                else:
                    ax.set_xlabel("Avalanche Duration", fontsize=13)
                    ax.set_title(f"Condition {cond}", fontsize=13)

                ax.set_ylabel("Probability Density", fontsize=13)
                ax.legend(fontsize=11)
                ax.grid(True, which="both", linestyle="--", alpha=0.5)

                plt.tight_layout()

            dict_results[cond] = dict_cond
        if plot == "size":
            fig_cond.suptitle("Avalanche Size Distribution", fontsize=15)
            # fig.suptitle("Avalanche Size Distribution over all trials")
        else:
            fig_cond.suptitle("Avalanche Duration Distribution", fontsize=15)
            # fig.suptitle("Avalanche Duration Distribution over all trials")

    return fig, fig_cond, dict_results