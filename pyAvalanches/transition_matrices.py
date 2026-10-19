from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import numpy as np
import os
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from .avalanches import TrialAvalanches

class ATMObj:
    """
    Stores the results of Avalanche Transition Matrix (ATM)
    computations for a single trial.

    Parameters
    ----------
    trial_id : int
        Unique identifier for the trial.
    avalanche_transition_matrices: List[np.ndarray]
        List of ATM for each avalanche
    symmetric: Boolean
        Indicates whether the ATMs have been symmetrized.
    method: String
        Specifies the method used to compute the ATM
    mean_atm: np.ndarray
        Mean ATM of the trial
    trial_label : int or str, optional
        Condition label for the trial. Defaults to None.
    """

    def __init__(
            self,
            trial_id: Union[int, str],
            avalanche_transition_matrices: List[np.ndarray],
            symmetric: bool,
            method: str,
            mean_atm: np.ndarray,
            trial_label: Optional[Union[int, str]] = None
    ):
        self.trial_id = trial_id
        self.avalanche_transition_matrices = avalanche_transition_matrices
        self.symmetric = symmetric  # whether ATMs are symmetrized or not
        self.method = method  # method used
        self.mean_atm = mean_atm
        self.trial_label = trial_label


@dataclass
class TrialATM:
    """
    Stores a collection of ATMObj instances, one per trial.

    Parameters
    ----------
    trials : list of ATMObj
        List of ATM computation results, one entry per trial.
        Defaults to an empty list.

    """

    def __init__(
            self,
            trials: Optional[List[ATMObj]] = None
    ):
        self.trials = trials if trials is not None else []

    def __repr__(self):
        return (
            f"TrialATMs(n_trials={len(self.trials)})"
        )

    @property
    def trial_count(self):
        """Get the total number of trials stored in the collection.

        Returns
        -------
        int
            Total number of trials currently stored.
        """
        return len(self.trials)

    @property
    def trial_labels(self):
        """Get condition labels for all trials in the collection.

        Returns
        -------
        list of str
            List of condition labels for all trials. Each element corresponds
            to the trial_label attribute of an ATMObj trial.
        """
        all_trials_labels = [trial.trial_label for trial in self.trials]
        return all_trials_labels

    @property
    def trial_indexes(self):
        """Get trial IDs for all trials in the collection.

        Returns
        -------
        list of int
            List of trial IDs for all trials. Each element corresponds
            to the trial_id attribute of an ATMObj trial.
        """
        all_trials_ids = [trial.trial_id for trial in self.trials]
        return all_trials_ids

    def add_trial(self, trial: ATMObj):
        """Add an ATMObj trial result to the collection.

        Parameters
        ----------
        trial : ATMObj
            The trial object to be added to the trials collection.
        """
        self.trials.append(trial)

    def get_trial_data(self, trial_idx):
        """Retrieve the ATMObj for a specific trial by its ID.

        Parameters
        ----------
        trial_idx : int
            The trial ID to retrieve.

        Returns
        -------
        ATMObj
            The ATM data object corresponding to the given trial ID.

        Raises
        ------
        ValueError
            If the specified trial ID is not found in the collection.
        """
        #trial = next((trial for trial in self.trials if trial.trial_id == trial_idx), None)

        for trial in self.trials:
            if int(trial.trial_id) == int(trial_idx):
                return trial

        #available_ids = [trial.trial_id for trial in self.trials]
        raise ValueError(
            f"Trial ID '{trial_idx}' not found. "
            #f"Available trial IDs are: {available_ids}"
        )


    def global_mean_atm(self) -> np.ndarray:
        """Compute the global mean Avalanche Transition Matrix (ATM) across all trials.

        Averages the Avalanche Transition Matrices from all trials to produce
        a single representative ATM.

        Returns
        -------
        np.ndarray
            Mean ATM of shape (n_channels, n_channels), averaged across all trials.

        Raises
        ------
        ValueError
            If the TrialATM collection is empty (contains no trials).
        """
        if self.trial_count == 0:
            raise ValueError("TrialATM object is empty")

        all_means = [trial.mean_atm for trial in self.trials]
        return np.mean(all_means, axis=0)

    def get_atm_mean_by_condition(self, cond) -> np.ndarray:
        """
        Computes the mean Avalanche Transition Matrix (ATM) across all trials
        matching a given condition label.

        Parameters
        ----------
        cond: int or str
            The condition label used to filter trials.

        Returns
        -------
        np.ndarray:
            ATM averaged across all trials matching the specified condition.
        """

        cond_means = [trial.mean_atm for trial in self.trials if trial.trial_label == cond]

        if len(cond_means) == 0:
            available_conditions = list(set(trial.trial_label for trial in self.trials))
            raise ValueError(
                f"Condition '{cond}' not found. "
                f"Available conditions are: {available_conditions}"
            )

        return np.mean(cond_means, axis=0)

    def save(self, out_path: str, prefix):
        """
        Serializes and saves the TrialATM object to a pickle file.

        Parameters
        ----------
        out_path : str
            Path to the directory where the file will be saved.
        prefix : str
            Prefix string used to name the output file. The file will be
            saved as `{prefix}_atm_obj.pkl`.
        """

        with open(os.path.join(out_path,f"{prefix}_atm_obj.pkl"),'wb') as f:
            pickle.dump(self, f)

class ATMPipeline(BaseEstimator, TransformerMixin):
    """
    Core pipeline for computing the Avalanche Transition
    Matrix (ATM) from neuronal avalanche data.

    The ATM captures the probability of transitions between spatial activation
    patterns across consecutive avalanches. Two computation methods are
    supported: a standard method and a weighted-sum (ws) method.

    - 'standard': Each element (i, j) is computed as the probability that
      region j was active at time t+1 given that region i was active at
      time t.
    - 'ws' (weighted-stochastic): The ATMs are computed as a weighted stochastic
      model following three steps:
        1. Transition counting — counting the number of consecutive
           activations between each pair of regions.
        2. Probability normalization — normalizing the transition counts
           into probabilities.
        3. Duration weighting — weighting the resulting matrices by the
           durations of their corresponding avalanches.


    Scikit-learn compatible.

    Parameters
    ----------
    method : str

        Method used to compute the ATM. Must be one of:
        - 'standard': computes the standard Avalanche Transition Matrix.
        - 'ws': computes the weighted-stochastic Avalanche Transition Matrix.
    symm : bool
        If True, the resulting ATM is symmetrized.
        This  parameter is ignored when using the "ws" method.
        Weighted-stochastic ATMs are inherently asymmetric and will always be
        returned without symmetrization.
    trial_label : int, optional
        If specified, only trials with this label are processed.
        Defaults to None.

    Raises
    ------
    ValueError
        If `method` is not one of the allowed methods ('standard', 'ws').
    """

    def __init__(
        self,
        method: str,
        symm: bool,
        trial_label: int = None # this is to select just a subset of trials based on their label (motor imagery, rest...)
    ):
        allowed_methods = {"ws", "standard"}

        if method not in allowed_methods:
            raise ValueError(
                f"method must be one of {allowed_methods}, got '{method}'"
            )

        self.method = method
        self.symm = symm
        self.trial_label = trial_label

    def __repr__(self):
        return (
            f"ATMPipeline(method={self.method}, "
            f"symm={self.symm})"
        )


    def _compute_atm_standard(self, bin_data: np.ndarray):
        """
        Compute the standard transition matrix of an avalanche.

        Parameters
        ----------
        bin_data : np.ndarray
            2D binary matrix of shape (n_regions, n_timepoints), where each element is 0 or 1,
            indicating whether a region is inactive or active at a given time.

        Returns
        -------
        transition_matrix : np.ndarray
            2D matrix of shape (n_regions, n_regions), where element (i, j) represents the
            normalized probability that the activation of region i at time t is followed by
            the activation of region j at time t+1.
        """
        num_regions, num_times = bin_data.shape
        ATM = np.zeros((num_regions, num_regions))
        num_activations_per_reg = np.sum(bin_data, axis = 1) # sum of binary data over regions -> shape: n_regions x 1

        for kk_reg in range(num_regions):
            reg_loop = np.zeros(num_regions) # this will count how many times each region activates right after region kk_reg
            for kk_time in range(num_times - 1):
                if bin_data[kk_reg, kk_time] == 1: # if region kk_reg is active at time kk_time
                    # Repeat each element of an array after themselves
                    curr_reg = np.repeat(bin_data[kk_reg, kk_time], num_regions)  # this creates a vector of 1s of length num_regions.
                    next_reg = bin_data[:, kk_time + 1] # activation pattern of all regions at time kk_time + 1
                    reg_tmp = curr_reg == next_reg # these regions are active at the next time step -> array of bool
                    reg_loop += reg_tmp # this counts how many times each region activates after kk_reg
            reg_loop[reg_loop != 0] /= num_activations_per_reg[kk_reg] # normalization
            ATM[kk_reg, :] = reg_loop

        if self.symm:
            symm_atm = (ATM + ATM.T) / 2  # Symmetrize
            return symm_atm
        else:
            return ATM

        # return ATM_list, mean_ATM

    def _compute_atm_ws(self, bin_data: np.ndarray):
        """
        Compute the transition matrix of an avalanche
        weighted by the size of the avalanche

        Parameters
        ----------
        bin_data : np.ndarray
            2D binary matrix of shape (n_regions, n_timepoints), where each element is 0 or 1,
            indicating whether a region is inactive or active at a given time.

        Returns
        -------
        transition_matrix : np.ndarray
            2D matrix of shape (n_regions, n_regions), where element (i, j) represents the
            normalized probability that the activation of region i at time t is followed by
            the activation of region j at time t+1.
        """

        num_regions, num_times = bin_data.shape

        # compute the transition counts -> calculate_counts di ws
        counts = np.zeros((num_regions, num_regions))
        for kk_time in range(num_times - 1):
            for kk_reg in range(num_regions): # If region `i` is active at time `t`
                if bin_data[kk_reg, kk_time] == 1:
                    for kk2_reg in range(num_regions):
                        if bin_data[kk2_reg, kk_time + 1] == 1: # If region `j` is active at time `t`
                            counts[kk_reg, kk2_reg] += 1

        # compute the stochastic ATM -> calculate_atm di ws
        ATM = np.zeros((num_regions, num_regions))
        e = np.ones((num_regions, 1))
        d = np.zeros((num_regions, 1))
        for i in range(num_regions):
            total = np.sum(counts[i,:])
            if total == 0:
                d[i] = 1
                ATM[i,:] = (1 / num_regions) * d[i] * e.T
            else:
                #print(f'counts {counts[i, :]} total {total}')
                ATM[i, :] = counts[i, :] / total

        return ATM

    def _compute_stochastic_mean_atm(self, trial_atm_data: ATMObj, avalanche_data):
        """
        Computes the weighted-stochastic mean ATM for a single trial, weighting each
        individual ATM by the duration of its corresponding avalanche.

        Parameters
        ----------
        trial_atm_data : ATMObj
            An ATMObj containing the list of individual Avalanche Transition
            Matrices for a single trial.
        avalanche_data : AvalancheObj
            An AvalancheObj containing the avalanche durations used as weights.

        Returns
        -------
        np.ndarray
            The duration-weighted mean Avalanche Transition Matrix for the trial.
        """
        # multiplying each ATM by the corresponding avalanche duration
        ws_atm_list = [ATM * duration
                     for ATM, duration in zip(trial_atm_data.avalanche_transition_matrices, avalanche_data.durations)]

        # averaging with the total duration
        ws_atm_mean = np.divide(
            np.sum(ws_atm_list, axis=0),
            np.sum(avalanche_data.durations)
        )

        # trial_atm.ws_ATM_mean = ws_ATM_mean

        return ws_atm_mean


    def process(self, avalanches: TrialAvalanches) -> TrialATM:
        """
        Runs the ATM computation pipeline on a TrialAvalanches object.

        Iterates through all trials (or a subset filtered by `self.trial_label`),
        computes the ATM for each avalanche using the specified method, and
        returns the results as a TrialATM object. For the weighted-sum method,
        the mean ATM is computed using duration-weighted averaging.

        Parameters
        ----------
        avalanches : TrialAvalanches
            A TrialAvalanches object containing the avalanche detection results
            for all trials.

        Returns
        -------
        TrialATM
            A TrialATM object containing one ATMObj per processed trial.

        Raises
        ------
        ValueError
            If `self.trial_label` is specified but no trials with that label
            are found in the input data.
        """

        if not isinstance(avalanches, TrialAvalanches):
            raise TypeError(f"'avalanches' must be a TrialAvalanches object, got {type(avalanches)} instead")

        list_results = []

        if self.trial_label is not None:
            # trial_indexes = np.where(dataset.trial_labels == self.trial_label)[0]
            #trial_indexes = [idx for idx, lab in enumerate(avalanches.trial_labels) if lab == self.trial_label] # -> errore qui, perché sto prendendo indice di lista e non id_trial
            trial_indexes = [id for id,label in zip(avalanches.trial_indexes, avalanches.trial_labels) if label == self.trial_label]

            # if trial_indexes.shape[0] == 0:
            if len(trial_indexes) == 0:
                raise ValueError(
                    f"No trials found with label {self.trial_label}"
                )
            else:
                print(f"Found {len(trial_indexes)} trials with label {self.trial_label}")

        else:
            # trial_indexes = range(dataset.n_trials)
            trial_indexes = avalanches.trial_indexes

        for trial_idx in trial_indexes: # number of trials
            print(f"Computing trial {trial_idx}...")
            trial_aval_data = avalanches.get_trial_data(trial_idx)
            trial_ATM_list = []

            if trial_aval_data.avalanche_count == 0:
                print("WARNING: no avalanches in this trial, no ATM was computed.")

            else:
                for aval_idx in range(trial_aval_data.avalanche_count):  # number of avalanches in that specific trial
                    #print(f"Computing ATM of trial {trial_idx}...")
                    bin_data = trial_aval_data.avalanches_mat[aval_idx]

                    if self.method == 'standard':
                        ATM = self._compute_atm_standard(bin_data)
                        trial_ATM_list.append(ATM)

                    if self.method == 'ws':
                        ATM = self._compute_atm_ws(bin_data)
                        trial_ATM_list.append(ATM)

                #mean_trial_atm = np.mean(trial_ATM_list, axis=0)

                # the mean doesn't have to take into account 0 values
                trial_ATM_array = np.stack(trial_ATM_list, axis=2)
                mean_trial_atm = np.sum(trial_ATM_array, axis=2) / np.sum(trial_ATM_array != 0, axis=2)
                mean_trial_atm[np.isnan(mean_trial_atm)] = 0

                trial_atm = ATMObj(trial_id=trial_idx, avalanche_transition_matrices=trial_ATM_list, method=self.method,
                                   symmetric=self.symm, mean_atm=mean_trial_atm)

                trial_atm.trial_label = trial_aval_data.trial_label

                if self.method == 'ws':
                    print('Computing weighted mean ATM')

                    trial_atm.mean_atm = self._compute_stochastic_mean_atm(trial_atm, trial_aval_data)
                    trial_atm.symmetric = False

                list_results.append(trial_atm)

            # qui finisce l'else

        results = TrialATM(list_results)
        print('Done')
        return results

    def fit(self, X, y=None):
        """
        Placeholder fit method for scikit-learn Pipeline compatibility.
        This method does not perform any computation and simply returns self.
        """
        # this doesn't need to do anything
        return self

    def transform(self, X: TrialAvalanches):
        """
        Runs the ATM computation pipeline and returns the results as a
        flattened 2D array for use within a scikit-learn Pipeline.

        Computes the mean ATM for each trial using `process()`, then reshapes
        the resulting 3D array of mean ATMs into a 2D array where each row
        corresponds to a flattened ATM matrix for a single trial.

        Warning
        ----------
        Currently the Scikit-learn compatibility ("fit"/"transform") is implemented
        solely to allow "ATMPipeline" to be used as a downstream step after
        "AvalanchePipeline" within a scikit-learn "Pipeline". It requires either
        being placed after "AvalanchePipeline" in a "Pipeline" or being directly
        fed an "AvalancheAnalysis" object. It is NOT designed to function as a
        standalone scikit-learn transformer and may not behave as expected
        outside of that context.

        Parameters
        ----------
        X : TrialAvalanches
            A TrialAvalanches object containing the avalanche detection results
            for all trials.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_trials, n_channels * n_channels) where each
            row is the flattened mean ATM for a single trial.

        """
        results_atm = self.process(X)
        print(results_atm.trial_count)
        dim_atm = np.shape(results_atm.get_trial_data(0).mean_atm)[0]
        atm_data = np.zeros([results_atm.trial_count, dim_atm, dim_atm])
        for i in range(0, results_atm.trial_count):
            atm_data[i, :, :] = results_atm.get_trial_data(i).mean_atm
        reshape_ATM = np.reshape(atm_data, (np.shape(atm_data)[0], np.shape(atm_data)[1] * np.shape(atm_data)[2]))

        return reshape_ATM