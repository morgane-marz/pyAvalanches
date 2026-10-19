from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, NamedTuple
import numpy as np
from scipy.stats import zscore, norm, skew
import os
import pickle
import networkx as nx
import bct
from .transition_matrices import TrialATM

class MetricsObj:
    """
        Stores the network metrics computed from the ATM of a single trial.

        Parameters
        ----------
        trial_id : int
            Unique identifier for the trial.
        atm_method : str
            The ATM computation method used (e.g., 'standard', 'ws').
        trial_label : int or str, optional
            Condition label for the trial (e.g., 'MI', 'rest'). Defaults to None.

        Note
        -----
        While not explicitly stated at the time of inizialization, MetricsObj
        will store as attributes the metrics selected to be computed.
        See the NetworkMetricsPipeline for more information on the metrics.

        """

    def __init__(
            self,
            trial_id: int,
            atm_method: str,
            trial_label: Optional[Union[int, str]] = None
    ):
        self.trial_id = trial_id
        self.atm_method = atm_method
        self.trial_label = trial_label  # could be MI, rest ...


class TrialMetrics:
    """
    Stores a collection of MetricsObj instances, one per trial.

    Parameters
    ----------
    trials : list of MetricsObj
        List of network metrics results, one entry per trial.
        Defaults to an empty list.
    """

    def __init__(
            self,
            trials: Optional[List[MetricsObj]] = None
    ):
        self.trials = trials if trials is not None else []

    def __repr__(self):
        return (
            f"TrialMetrics(n_trials={len(self.trials)})"
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
            to the trial_label attribute of an MetricsObj trial.
        """
        # all_trials_labels = []
        # for idx in range(len(self.trials)):
        #    all_trials_labels.append(self.trials[idx].trial_label)
        all_trials_labels = [trial.trial_label for trial in self.trials]
        return all_trials_labels

    @property
    def trial_indexes(self):
        """Get trial IDs for all trials in the collection.

        Returns
        -------
        list of int
            List of trial IDs for all trials. Each element corresponds
            to the trial_id attribute of an MetricsObj trial.
        """
        # all_trials_ids = []
        # for idx in range(len(self.trials)):
        #    all_trials_ids.append(self.trials[idx].trial_id)
        all_trials_ids = [trial.trial_id for trial in self.trials]
        return all_trials_ids

    def get_trial_data(self, trial_idx):
        """Retrieve the MetricsObj for a specific trial by its ID.

        Parameters
        ----------
        trial_idx : int
            The trial ID to retrieve.

        Returns
        -------
        MetricsObj
            The metrics data object corresponding to the given trial ID.

        Raises
        ------
        ValueError
            If the specified trial ID is not found in the collection.
        """
        # return self.trials[trial_idx]
        #return next(trial for trial in self.trials if trial.trial_id == trial_idx)
        for trial in self.trials:
            if int(trial.trial_id) == int(trial_idx):
                return trial

        #available_ids = [trial.trial_id for trial in self.trials]
        raise ValueError(
            f"Trial ID '{trial_idx}' not found. "
            #f"Available trial IDs are: {available_ids}"
        )

    def save(self, out_path, prefix):
        """
        Serializes and saves the TrialMetrics object to a pickle file.

        Parameters
        ----------
        out_path : str
            Path to the directory where the file will be saved.
        prefix : str
            Prefix string used to name the output file. The file will be
            saved as `{prefix}_metrics_obj.pkl`.
        """

        with open(os.path.join(out_path, f"{prefix}_metrics_obj.pkl"), 'wb') as f:
            pickle.dump(self, f)

class NetworkMetricsPipeline:
    """
    Pipeline for computing graph-theory based network metrics from
    Avalanche Transition Matrices (ATMs).

    Supports both symmetric (undirected) and non-symmetric (directed) ATMs,
    with different sets of available metrics for each case.

    Parameters
    ----------
    metrics : list of str
        List of metrics to compute. For symmetric ATMs, allowed values are:
        'clustering_coeff', 'node_strength', 'eigenvect_centrality',
        'modularity', 'char_path', 'connectivity_measures'.
        For non-symmetric ATMs, only 'connectivity_measures' is allowed.
    symm : bool
        If True, the pipeline expects symmetric (undirected) ATMs.
        If False, the pipeline expects non-symmetric (directed) ATMs.

    Raises
    ------
    ValueError
        If any metric in `metrics` is not valid for the specified
        symmetry setting.
    """

    def __init__(
            self,
            metrics: List[str],
            symm: bool
    ):

        if len(metrics) == 0:
            raise ValueError(
                "Empty metrics list was given - please choose at least one metric to compute"
            )

        if not isinstance(symm, (bool, np.bool_)):
            raise TypeError(
                f"'symmetric' must be a boolean, got {symm!r} "
                f"({type(symm).__name__})"
            )
        #self.symmetric = bool(symm)

        if symm:
            allowed_metrics = {'clustering_coeff', 'node_strength', 'eigenvect_centrality',
                               'modularity', 'char_path', 'connectivity_measures'}
            # check if input metrics are okay
            for input_metric in metrics:
                if input_metric not in allowed_metrics:
                    raise ValueError(
                        f"for undirected (symmetric) ATMs, metric must be one of {allowed_metrics}, got '{input_metric}'"
                    )

            self.metrics = metrics
            self.symm = symm

        else:
            allowed_metrics = {'connectivity_measures'}
            # check if input metrics are okay
            for input_metric in metrics:
                if input_metric not in allowed_metrics:
                    raise ValueError(
                        f"for directed (non-symmetric) ATMs, metric must be one of {allowed_metrics}, got '{input_metric}'"
                    )

            self.metrics = metrics
            self.symm = symm

    def __repr__(self):
        return f"NetworkMetricsPipeline(metrics={self.metrics}"

    @staticmethod
    def _compute_clust_coeff(matrix: np.ndarray):
        """
        Computes the weighted clustering coefficient for each node.

        Parameters
        ----------
        matrix : np.ndarray
          ATM of shape (n_regions, n_regions).

        Returns
        -------
        clustering_coefficient : np.ndarray
          Clustering coefficient for each node, shape (n_nodes,).
        avg_clustering_coefficient : float
          Mean clustering coefficient across all nodes.
        """
        clustering_coefficient = bct.clustering_coef_wu(matrix)
        avg_clustering_coefficient = np.mean(clustering_coefficient)

        return clustering_coefficient, avg_clustering_coefficient

    @staticmethod
    def _compute_node_strength(matrix: np.ndarray):
        """
        Computes the strength of each node in the network.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        node_strength : np.ndarray
            Strength (sum of edge weights) for each node, shape (n_nodes,).
        avg_node_strength : float
            Mean node strength across all nodes.
        """

        node_strength = bct.strengths_und(matrix)  # shape: (n_nodes,)
        avg_node_strength = np.mean(node_strength)

        return node_strength, avg_node_strength

    @staticmethod
    def _compute_eigen_centr(matrix: np.ndarray):
        """
        Computes the eigenvector centrality for each node in the network.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        eigenvector_centrality : np.ndarray
            Eigenvector centrality for each node, shape (n_nodes,).
        avg_eigenvector_centrality : float
            Mean eigenvector centrality across all nodes.
        """
        G = nx.from_numpy_array(matrix)
        eigenvector_centrality = np.asarray(list(nx.eigenvector_centrality_numpy(G, weight='weight').values()))
        avg_eigenvector_centrality = np.mean(eigenvector_centrality)

        return eigenvector_centrality, avg_eigenvector_centrality

    @staticmethod
    def _compute_modularity(matrix: np.ndarray):
        """
        Computes the modularity of the network using label propagation
        to detect communities.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        modularity : float
            Modularity score of the detected community partition.
        """
        G = nx.from_numpy_array(matrix)
        communities = list(nx.algorithms.community.label_propagation_communities(G))
        modularity = nx.algorithms.community.modularity(G, communities)

        return modularity

    @staticmethod
    def _compute_charpath_len(matrix: np.ndarray):
        """
        Computes the characteristic path length of the network.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        charpath_len : float
            Characteristic path length of the network.
        """
        charpath_len = bct.charpath(matrix)[0]
        return charpath_len

    @staticmethod
    def _compute_connectivity_measures_orig(matrix: np.ndarray):
        """
        Computes basic connectivity statistics of the ATM.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        mean_mat : float
            Mean of the matrix entries.
        var_mat : float
            Variance of the matrix entries.
        var_coeff : float
            Coefficient of variation (mean divided by variance).
        skew_mat : np.ndarray
            Skewness of the matrix computed along each column.
        """
        mean_mat = np.mean(matrix)
        var_mat = np.var(matrix)
        var_coeff = mean_mat / var_mat
        skew_mat = skew(matrix)

        return mean_mat, var_mat, var_coeff, skew_mat

    @staticmethod
    def _compute_connectivity_measures(matrix: np.ndarray):
        """
        Computes basic connectivity statistics of the ATM.

        Parameters
        ----------
        matrix : np.ndarray
            ATM of shape (n_regions, n_regions).

        Returns
        -------
        mean_mat : float
            Mean of the matrix entries.
        var_mat : float
            Variance of the matrix entries.
        var_coeff : float
            Coefficient of variation (standard deviation divided by the mean).
            Returns np.nan when the mean is zero, since the ratio is undefined.
        skew_mat : np.ndarray
            Skewness of the matrix computed along each column.

        Notes
        -----
        The coefficient of variation is only meaningful for non-negative data
        with a non-zero mean, which holds for ATMs (all entries are
        probabilities in [0, 1]).
        """
        mean_mat = float(np.mean(matrix))
        var_mat = float(np.var(matrix))
        std_mat = float(np.std(matrix))

        # An all-zero ATM has mean 0, making the ratio undefined.
        # np.nan propagates visibly instead of silently producing inf.
        if mean_mat == 0:
            var_coeff = np.nan
        else:
            var_coeff = std_mat / mean_mat

        skew_mat = skew(matrix)

        return mean_mat, var_mat, var_coeff, skew_mat

    def process(self, atm_data: TrialATM):
        """
        Runs the network metrics computation pipeline on a TrialATM object.

        Filters trials based on the expected symmetry setting, then computes
        the requested network metrics for each valid trial's mean ATM.
        Results are stored in MetricsObj instances and returned as a
        TrialMetrics object.

        Parameters
        ----------
        atm_data : TrialATM
            A TrialATM object containing the ATM results for all trials.

        Returns
        -------
        TrialMetrics
            A TrialMetrics object containing one MetricsObj per processed
            trial, with the requested network metrics stored as attributes.

        Raises
        ------
        ValueError
            If no ATMs matching the expected symmetry setting are found
            in the input data.
        """

        # check for empty input
        n_input = len(atm_data.trial_indexes)
        if n_input == 0:
            raise ValueError("'atm_data' contains no trials")

        expected = bool(self.symm)
        kind = "symmetric" if expected else "non-symmetric"

        checked_trials = []
        for trial_id in atm_data.trial_indexes:
            trial = atm_data.get_trial_data(trial_id)

            if not isinstance(trial.symmetric, (bool, np.bool_)):
                raise TypeError(
                    f"trial {trial_id}: 'symmetric' must be a boolean, "
                    f"got {trial.symmetric!r} ({type(trial.symmetric).__name__})"
                )

            if bool(trial.symmetric) == expected:
                checked_trials.append(trial)

        n_checked = len(checked_trials)
        if n_checked == 0:
            raise ValueError(f"No {kind} ATMs found in the input")
        elif n_checked == n_input:
            print(f"All ATMs are {kind}, computing network metrics...")
        else:
            print(f"Found {n_checked} {kind} ATMs on a total of {n_input}, "
                  f"computing metrics for those...")

        """
        if self.symm:  # if symm == True

            checked_ids = []
            for i in atm_data.trial_indexes:
                if atm_data.get_trial_data(i).symmetric:  # if True
                    checked_ids.append(i)

            if len(checked_ids) == len(atm_data.trial_indexes):
                print("All ATMs are symmetric, computing network metrics...")
            elif len(checked_ids) < len(atm_data.trial_indexes) and len(checked_ids) != 0:
                print(
                    f"Found {len(checked_ids)} symmetric ATMs on a total of {len(atm_data.trial_indexes)}, computing metrics for those...")
            elif len(checked_ids) == 0:
                raise ValueError(
                    "No symmetrical ATMs found in the input"
                )
            checked_labels = [atm_data.get_trial_data(i).trial_label for i in checked_ids]

        else:
            checked_ids = []
            for i in atm_data.trial_indexes:
                if atm_data.get_trial_data(i).symmetric == False:
                    checked_ids.append(i)

            if len(checked_ids) == len(atm_data.trial_indexes):
                print("All ATMs are non-symmetric, computing network metrics...")
            elif len(checked_ids) < len(atm_data.trial_indexes) and len(checked_ids) != 0:
                print(
                    f"Found {len(checked_ids)} non-symmetric ATMs on a total of {len(atm_data.trial_indexes)}, computing metrics for those...")
            elif len(checked_ids) == 0:
                raise ValueError(
                    "No non-symmetrical ATMs found in the input"
                )
            checked_labels = [atm_data.get_trial_data(i).trial_label for i in checked_ids]
        """

        list_metrics_trials = []

        #for trial_idx, label in zip(checked_ids, checked_labels):

            #for trial_idx, label in zip(atm_data.trial_indexes, atm_data.trial_labels):
        #    print(f"Computing trial {trial_idx}...")

        #    atm_method = atm_data.get_trial_data(trial_idx).method
        #    matrix = atm_data.get_trial_data(trial_idx).mean_atm

        #    metrics_obj = MetricsObj(trial_id=trial_idx, atm_method= atm_method,trial_label=label)

        for trial in checked_trials:
            print(f"Computing trial {trial.trial_id}...")

            matrix = trial.mean_atm
            metrics_obj = MetricsObj(
                trial_id=trial.trial_id,
                atm_method=trial.method,
                trial_label=trial.trial_label,
            )

            if 'connectivity_measures' in self.metrics:
                mean_mat, var_mat, var_coeff, skew_mat = self._compute_connectivity_measures(matrix)
                metrics_obj.mean_mat = mean_mat
                metrics_obj.var_mat = var_mat
                metrics_obj.var_coeff = var_coeff
                metrics_obj.skew_mat = skew_mat

            if 'clustering_coeff' in self.metrics:
                clustering_coefficient, avg_clustering_coefficient = self._compute_clust_coeff(matrix)
                metrics_obj.clustering_coefficient = clustering_coefficient
                metrics_obj.avg_clustering_coefficient = avg_clustering_coefficient

            if 'node_strength' in self.metrics:
                node_strength, avg_node_strength = self._compute_node_strength(matrix)
                metrics_obj.node_strength = node_strength
                metrics_obj.avg_node_strength = avg_node_strength

            if 'eigenvect_centrality' in self.metrics:
                eigenvector_centrality, avg_eigenvector_centrality = self._compute_eigen_centr(matrix)
                metrics_obj.eigenvector_centrality = eigenvector_centrality
                metrics_obj.avg_eigenvector_centrality = avg_eigenvector_centrality

            if 'modularity' in self.metrics:
                modularity = self._compute_modularity(matrix)
                metrics_obj.modularity = modularity

            if 'char_path' in self.metrics:
                char_path = self._compute_charpath_len(matrix)
                metrics_obj.char_path = char_path

            list_metrics_trials.append(metrics_obj)

        results = TrialMetrics(list_metrics_trials)
        return results