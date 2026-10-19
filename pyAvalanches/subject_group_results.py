from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, NamedTuple
import numpy as np
import pandas as pd
import os
import pickle
import mne

from .avalanches import TrialAvalanches
from .transition_matrices import TrialATM
from .dataset import Dataset, create_dataset_from_mne
from .connectivity_metrics import TrialMetrics

# helpers shared by Subject and GroupResults dataframe builders

def _sorted_conditions(labels) -> list:
    """
    Unique condition labels in a deterministic order.
    """
    unique = set(labels)
    has_none = None in unique
    unique.discard(None)
    ordered = sorted(unique, key=str)
    if has_none:
        ordered.append(None)
    return ordered

def _group_trials_by_condition(container) -> dict:
    """
    Groups a trial container's entries by their trial_label.

    Parameters
    ----------
    container : TrialAvalanches, TrialATM or TrialMetrics
        Any container exposing a `.trials` list.

    Returns
    -------
    dict
        {condition: [trial objects]} with conditions in deterministic order
        and trials in their original order within each condition.
    """

    grouped = dict()
    for trial in container.trials:
        grouped.setdefault(trial.trial_label, []).append(trial)
    return {cond: grouped[cond] for cond in _sorted_conditions(grouped)}

def _na(value):
    """Converts None to pd.NA so missing conditions render consistently."""
    return pd.NA if value is None else value

# attributes exposed as fixed columns, so excluded from dynamic capture.
_METRICS_EXCLUDED_ATTRS = frozenset({"trial_id", "trial_label", "atm_method"})

def _metric_items(metrics_obj):
    """
    Yields (column_name, value) for every public metric on a MetricsObj.

    Column names are snake_case converted to Title Case:
        node_strength -> 'Node Strength'
    """
    for attr, value in vars(metrics_obj).items():
        if attr.startswith("_") or callable(value):
            continue
        if attr in _METRICS_EXCLUDED_ATTRS:
            continue
        yield attr.replace("_", " ").title(), value

def _mean_metric(values: list):
    """
    Averages a list of metric values.

    Array-valued metrics (one entry per node) are averaged element-wise so
    the per-node shape is preserved; scalars are averaged to a scalar.
    """
    if isinstance(values[0], (list, np.ndarray)):
        return np.mean(np.asarray(values), axis=0)
    return float(np.mean(values))

_DATASET_EXCLUDED = frozenset({"channel_names"})

#: Maps internal dataset keys to report column names, in output order.
_DATASET_RENAME = {
    "fmin":          "Highpass",
    "fmax":          "Lowpass",
    "n_channels":    "Number of Channels/ROIs",
    "n_time_points": "Time Points",
    "n_trials":      "Number of Trials",
    "samp_freq":     "Sampling Frequency",
    "conditions":    "Trial Conditions",
}

def _dataset_row(dataset: dict) -> dict:
    """
    Turns a processed dataset info dict into a report row.
    """
    if dataset is None:
        raise ValueError("dataset metadata is not available")

    row = {}
    for key, column in _DATASET_RENAME.items():
        if key in dataset:
            row[column] = dataset[key]
    # forward any extra keys unchanged, minus the excluded ones
    for key, value in dataset.items():
        if key not in _DATASET_RENAME and key not in _DATASET_EXCLUDED:
            row[key] = value
    return row

def _require(value, name: str, owner: str):
    """
    Returns `value`, or raises a clear error if it is None.
    """
    if value is None:
        raise ValueError(
            f"{owner}: '{name}' is not available - run that analysis first"
        )
    return value

# column order for subject dataframes
AVALANCHE_COLUMNS_SUBJECT = [
    "Subject ID", "Trial ID", "Condition", "Aval Count", "Aval Sizes",
    "Aval Durations", "Flexibility", "Patterns", "Temporal Patterns",
]

PATTERNS_COLUMNS_SUBJECT = ["Subject ID", "Trial ID", "Condition", "Patterns", "Temporal Patterns"]

ATM_COLUMNS_SUBJECT = ["Subject ID", "Trial ID", "Condition", "ATM Method", "Mean ATM"]

METRICS_BASE_COLUMNS_SUBJECT = ["Subject ID", "Trial ID", "Condition", "ATM Method"]

AVALANCHE_COLUMNS_GROUP = [
    "Group", "Subject ID", "Condition", "Aval Sizes", "Aval Durations",
    "Mean Flexibility", "Mean Count",
]

PATTERNS_COLUMNS_GROUP = [
    "Group", "Subject ID", "Condition", "Trial ID",
    "Patterns", "Temporal Patterns",
]

ATM_COLUMNS_GROUP = ["Group", "Subject ID", "Condition", "ATM Method", "Mean ATM",]

METRICS_BASE_COLUMNS_GROUP = ["Group", "Subject ID", "Condition", "ATM Method"]

DATASET_ID_COLUMNS_GROUP = ["Group", "Subject ID"]

# metrics kept by get_averaged_metrics_dataframe(): one value per node.
NODE_LEVEL_METRICS = (
        "clustering_coefficient",
        "node_strength",
        "eigenvector_centrality",
    )


@dataclass
class Subject:
    """
    Stores all analysis results and metadata for a single subject.

    Upon initialization, if a Dataset object is provided, it is replaced
    by a lightweight dictionary containing only the dataset metadata
    (e.g., number of channels, trials, sampling frequency), discarding
    the raw time series data to reduce memory usage.

    Parameters
    ----------
    subject_id : int or str
        Unique identifier for the subject.
    subject_label : int or str, optional
        Group label for the subject (e.g., patient, control).
        Defaults to None.
    avalanche_results : TrialAvalanches, optional
        Avalanche detection results for all trials. Defaults to None.
    atm_results : TrialATM, optional
        Avalanche Transition Matrix results for all trials.
        Defaults to None.
    metrics_results : TrialMetrics, optional
        Network metrics results for all trials. Defaults to None.
    dataset : Dataset or dict, optional
        If a Dataset object is provided at initialization, it is
        converted to a dictionary containing the following keys:
        'n_channels', 'n_time_points', 'n_trials', 'conditions',
        'channel_names', 'samp_freq', 'fmax', 'fmin'.
        Defaults to None.
    """
    subject_id: Union[int, str]
    subject_label: Optional[Union[int, str]] = None
    avalanche_results: Optional[TrialAvalanches] = None
    atm_results: Optional[TrialATM] = None
    metrics_results: Optional[TrialMetrics] = None
    #dataset: Optional[Dataset] = None
    dataset: Optional[Union[mne.io.BaseRaw, mne.BaseEpochs, dict]] = None

    def __post_init__(self):
        #TODO edit here
        if self.dataset is not None:

            if isinstance(self.dataset, (mne.io.BaseRaw, mne.BaseEpochs)):
                dataset_obj = create_dataset_from_mne(self.dataset)

                dataset_info = dict()
                dataset_info['n_channels'] = dataset_obj.n_channels
                dataset_info['n_time_points'] = dataset_obj.n_time_points
                dataset_info['n_trials'] = dataset_obj.n_trials
                if dataset_obj.trial_labels is not None:
                    dataset_info['conditions'] = set(dataset_obj.trial_labels)
                else:
                    dataset_info['conditions'] = None
                dataset_info['channel_names'] = dataset_obj.channel_names
                dataset_info['samp_freq'] = dataset_obj.samp_freq
                dataset_info['fmax'] = dataset_obj.fmax
                dataset_info['fmin'] = dataset_obj.fmin

            elif isinstance(self.dataset, dict):
                dataset_info = {}
                dataset_info['n_channels'] = self.dataset["data_array"].shape[1]
                dataset_info['n_time_points'] = self.dataset["data_array"].shape[2]
                dataset_info['n_trials'] = self.dataset["data_array"].shape[0]
                if self.dataset["trial_labels"] is not None:
                    dataset_info['conditions'] = set(self.dataset["trial_labels"])
                else:
                    dataset_info['conditions'] = None
                dataset_info['channel_names'] = self.dataset["channel_names"]
                dataset_info['samp_freq'] = self.dataset["samp_freq"]
                dataset_info['fmax'] = self.dataset["fmax"]
                dataset_info['fmin'] = self.dataset["fmin"]

            else:
                raise TypeError("Dataset input must be Raw or Epochs object or dict")

            self.dataset = dataset_info


    def get_mean_atm_cond(self, cond) -> np.ndarray:
        """
        Mean ATM across all trials matching a condition.

        Raises
        ------
        ValueError
            If no trial carries the requested condition (previously this
            returned NaN with a RuntimeWarning).
        """
        results = _require(self.atm_results, "atm_results",
                           f"Subject {self.subject_id}")

        matrices = [t.mean_atm for t in results.trials if t.trial_label == cond]
        if not matrices:
            available = _sorted_conditions(t.trial_label for t in results.trials)
            raise ValueError(
                f"Subject {self.subject_id}: no trials with condition {cond!r}. "
                f"Available: {available}"
            )
        return np.mean(matrices, axis=0)

    def get_avalanche_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of avalanche results, one row per trial.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Subject ID', 'Trial ID', 'Condition', 'Aval Count',
            'Aval Sizes', 'Aval Durations', 'Flexibility', 'Patterns',
            'Temporal Patterns'.

        Raises
        ------
        ValueError
            If avalanche results have not been computed.
        """

        results = _require(self.avalanche_results, "avalanche_results",
                           f"Subject {self.subject_id}")

        rows = [
            {
                "Subject ID": self.subject_id,
                "Trial ID": trial.trial_id,
                "Condition": _na(trial.trial_label),
                "Aval Count": trial.avalanche_count,
                "Aval Sizes": trial.sizes,
                "Aval Durations": trial.durations,
                "Flexibility": trial.flexibility,
                "Patterns": trial.patterns,
                "Temporal Patterns": trial.patterns_temporal,
            }
            for trial in results.trials
        ]
        # columns= keeps the schema even when rows is empty
        return pd.DataFrame(rows, columns=AVALANCHE_COLUMNS_SUBJECT)

    def get_patterns_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of spatial and temporal patterns, one row per trial.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Subject ID', 'Trial ID', 'Condition', 'Patterns',
            'Temporal Patterns'.
        """
        results = _require(self.avalanche_results, "avalanche_results",
                           f"Subject {self.subject_id}")

        rows = [
            {
                "Subject ID": self.subject_id,
                "Trial ID": trial.trial_id,
                "Condition": _na(trial.trial_label),
                "Patterns": trial.patterns,
                "Temporal Patterns": trial.patterns_temporal,
            }
            for trial in results.trials
        ]
        return pd.DataFrame(rows, columns=PATTERNS_COLUMNS_SUBJECT)

    def get_atm_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of mean ATMs, one row per trial.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Subject ID', 'Trial ID', 'Condition', 'ATM Method',
            'Mean ATM'. Each 'Mean ATM' cell holds a 2-D numpy array.
        """
        results = _require(self.atm_results, "atm_results",
                           f"Subject {self.subject_id}")

        rows = [
            {
                "Subject ID": self.subject_id,
                "Trial ID": trial.trial_id,
                "Condition": _na(trial.trial_label),
                "ATM Method": trial.method,  # read per trial, not leaked
                "Mean ATM": trial.mean_atm,
            }
            for trial in results.trials
        ]
        return pd.DataFrame(rows, columns=ATM_COLUMNS_SUBJECT)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of network metrics, one row per trial.

        Base columns are always present; one further column is added per
        metric found on the MetricsObj, named by converting the attribute
        from snake_case to Title Case.

        Returns
        -------
        pandas.DataFrame
            Base columns 'Subject ID', 'Trial ID', 'Condition', 'ATM Method'
            plus one column per computed metric.
        """
        results = _require(self.metrics_results, "metrics_results",
                           f"Subject {self.subject_id}")

        rows = []
        for trial in results.trials:
            row = {
                "Subject ID": self.subject_id,
                "Trial ID": trial.trial_id,
                "Condition": _na(trial.trial_label),
                "ATM Method": trial.atm_method,
            }
            row.update(_metric_items(trial))
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=METRICS_BASE_COLUMNS_SUBJECT)

        df = pd.DataFrame(rows)
        # Force base columns first; metric columns keep discovery order.
        metric_cols = [c for c in df.columns if c not in METRICS_BASE_COLUMNS_SUBJECT]
        return df[METRICS_BASE_COLUMNS_SUBJECT + metric_cols]

    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Builds a single-row DataFrame of this subject's dataset metadata.
        'channel_names' is excluded.

        Returns
        -------
        pandas.DataFrame
            One row: 'Subject ID' followed by the renamed metadata columns.
        """
        dataset = _require(self.dataset, "dataset", f"Subject {self.subject_id}")

        row = {"Subject ID": self.subject_id, **_dataset_row(dataset)}
        return pd.DataFrame([row])


    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Builds a comprehensive DataFrame with all available analysis data for
        this subject, one row per trial. Missing result types are skipped.

        Returns
        -------
        pandas.DataFrame
            Columns include: 'Subject ID', 'Trial ID', 'Condition', plus
            whatever columns are available from: avalanche data, ATM data,
            metrics, and dataset metadata.

        Raises
        ------
        ValueError
            If at least one result type (avalanche, atm, or metrics) is not
            computed, or dataset is not available.
        """
        # Get available data
        aval_results = self.avalanche_results
        atm_results = self.atm_results
        metrics_results = self.metrics_results
        dataset = self.dataset
        dataset_metadata = _dataset_row(dataset) if dataset else {}

        # Determine which results are available and use the first non-None as trial source
        available_results = [r for r in [aval_results, atm_results, metrics_results] if r is not None]

        if not available_results:
            return pd.DataFrame()

        trial_source = available_results[0]

        rows = []
        for trial in trial_source.trials:
            row = {
                "Subject ID": self.subject_id,
                "Trial ID": trial.trial_id,
                "Condition": _na(getattr(trial, 'trial_label', None)),
            }

            # Avalanche data
            if aval_results:
                aval_trial = next((t for t in aval_results.trials if t.trial_id == trial.trial_id), trial)
                row.update({
                    "Avalanche Count": aval_trial.avalanche_count,
                    "Mean Size": np.mean(aval_trial.sizes),
                    "STD Size": np.std(aval_trial.sizes),
                    "Min Size": min(aval_trial.sizes),
                    "Max Size": max(aval_trial.sizes),
                    "Mean Duration": np.mean(aval_trial.durations),
                    "STD Duration": np.std(aval_trial.durations),
                    "Min Duration": min(aval_trial.durations),
                    "Max Duration": max(aval_trial.durations),
                    "Flexibility": aval_trial.flexibility,
                })

            # ATM data
            if atm_results:
                atm_trial = next((t for t in self.atm_results.trials if t.trial_id == trial.trial_id), trial)
                row.update({
                    "ATM Method": atm_trial.method,
                    "Mean ATM Value": np.mean(atm_trial.mean_atm),
                })

            # Metrics data
            #if metrics_results:
            #    metrics_trial = next((t for t in metrics_results.trials if t.trial_id == trial.trial_id), trial)
            #    row.update(_metric_items(metrics_trial))

            # Dataset metadata
            row.update(dataset_metadata)

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def save(self, out_path, prefix):

        """
        Saves the subject's results to disk as CSV files and a pickle file,
        and generates a report.

        Saves whichever results are available (avalanche results, ATM results,
        metrics results) as separate CSV files, generates a summary report,
        and serializes the entire Subject object as a pickle file.

        Parameters
        ----------
        out_path : str
            Path to the directory where all output files will be saved.
        prefix : str
            Prefix string used to name all output files.

        Notes
        -----
        The following files are saved depending on available results:

        - ``{prefix}_avalanche_results.csv``
        - ``{prefix}_avalanche_patterns_results.csv``
        - ``{prefix}_atm_results.csv``
        - ``{prefix}_metrics_results.csv``
        - ``{prefix}_subject_results.pkl``
        """

        #if self.avalanche_results is not None:
        #    df_aval = self.get_avalanche_dataframe()
        #    df_aval.to_csv(os.path.join(out_path, f"{prefix}_avalanche_results.csv"), index=False)

        #    df_pattern = self.get_patterns_dataframe()
        #    df_pattern.to_csv(os.path.join(out_path, f"{prefix}_avalanche_patterns_results.csv"), index=False)

        #if self.atm_results is not None:
        #    df_atm = self.get_atm_dataframe()
        #    df_atm.to_csv(os.path.join(out_path, f"{prefix}_atm_results.csv"), index=False)

        #if self.metrics_results is not None:
        #    df_metrics = self.get_metrics_dataframe()
        #    df_metrics.to_csv(os.path.join(out_path, f"{prefix}_metrics_results.csv"), index=False)

        #report = generate_report(results=self, out_path=out_path, prefix=prefix)

        df_subj = self.get_summary_dataframe()
        df_subj.to_csv(os.path.join(out_path, f"{prefix}_subject_results.csv"), index=False)

        with open(os.path.join(out_path, f"{prefix}_subject_results.pkl"), 'wb') as f:
            pickle.dump(self, f)

@dataclass
class GroupResults:
    """
    Stores analysis results for a group of subjects, providing aggregation
    and access utilities across all subjects.

    Parameters
    ----------
    subjects : list of Subject
        List of Subject objects belonging to the group.
        Defaults to an empty list.
    channel_names : list of str, optional
        Names of the channels shared across all subjects. Defaults to None.
    info : dict, optional
        Additional metadata about the group. Defaults to None.

    Attributes
    ----------
    subject_ids : list of int
        List of unique identifiers for all subjects in the group.
    subject_labels : list of int
        List of group labels for all subjects in the group.
    n_subjects : int
        Total number of subjects in the group.
    """

    subjects: List[Subject] = field(default_factory=list)
    channel_names: Optional[List[str]] = None
    info: Optional[dict] = None

    def add_subject(self, subject: Subject) -> None:
        self.subjects.append(subject)

    def get_subject(self, subject_id: Union[int, str]) -> Subject:

        for subj in self.subjects:
            if subj.subject_id == subject_id:
                return subj

        #available_ids = [trial.trial_id for trial in self.trials]
        raise ValueError(
            f"Subject ID '{subject_id}' not found. "
            #f"Available trial IDs are: {available_ids}"
        )
        #return next((s for s in self.subjects if s.subject_id == subject_id), None)

    @property
    def subject_ids(self) -> List[int]:
        all_subj_ids = [subj.subject_id for subj in self.subjects]
        return all_subj_ids

    @property
    def subject_labels(self) -> List[int]:
        all_subj_labels = [subj.subject_label for subj in self.subjects]
        return all_subj_labels

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    def get_avalanche_subj(self, idx):
        """
        Retrieves the avalanche results for a specific subject
        """
        # if we have just one trial, returns automatically the single trial results
        if self.get_subject(idx).avalanche_results.trial_count == 1:
            return self.get_subject(idx).avalanche_results.trials[0]
        return self.get_subject(idx).avalanche_results

    def get_atm_subj(self, idx):
        """
        retrieves the atm results for a specific subject
        """
        # if we have just one trial, returns automatically the single trial results
        if self.get_subject(idx).atm_results.trial_count == 1:
            return self.get_subject(idx).atm_results.get_trial_data(0)
        return self.get_subject(idx).atm_results

    def get_metrics_subj(self, idx):
        """
        retrieves the metrics results for a specific subject
        """
        # if we have just one trial, returns automatically the single trial results
        if self.get_subject(idx).metrics_results.trial_count == 1:
            return self.get_subject(idx).metrics_results.get_trial_data(0)
        return self.get_subject(idx).metrics_results

    def get_mean_subj_atm(self, idx) -> np.ndarray:
        # computes the global mean Avalanche Transition Matrix (ATM) across
        # all trials for a specific subject.
        atm_obj = self.get_subject(idx).atm_results
        mean_atm = atm_obj.global_mean_atm()
        return mean_atm

    def all_subjects_have(self, attr: str) -> bool:
        """Return True if the list is non-empty and every subject has
        the named attribute populated (i.e. not None).

        Parameters
        ----------
        attr : str
            Name of the Subject attribute to check
            (e.g. 'avalanche_results', 'atm_results', 'metrics_results').
        """
        return bool(self.subjects) and all(
            getattr(s, attr, None) is not None for s in self.subjects
        )

    def get_mean_atm_group(self, group, cond) -> np.ndarray:
        """
        Mean ATM across all subjects with a given group label, for one
        condition.

        Raises
        ------
        ValueError
            If no subject carries the requested group label.
        """
        matrices = [
            subject.get_mean_atm_cond(cond=cond)
            for subject in self.subjects
            if subject.subject_label == group
        ]
        if not matrices:
            available = _sorted_conditions(self.subject_labels)
            raise ValueError(
                f"No subjects with label {group!r}. Available: {available}"
            )
        return np.mean(matrices, axis=0)


    def extract_group(self, label) -> "GroupResults":
        """
        Returns a new GroupResults holding only the subjects with `label`.
        Subjects are shared by reference, not copied.
        """
        matching = [s for s in self.subjects if s.subject_label == label]
        return GroupResults(subjects=matching,
                            channel_names=self.channel_names,
                            info=self.info)

    # iteration internal helper
    def _iter_subjects(self, attr: str = None):
        """
        Yields subjects in insertion order.

        Parameters
        ----------
        attr : str, optional
            If given, subjects whose `attr` is None are skipped, so a group
            with partial results still produces a usable frame instead of
            raising.
        """
        for subject in self.subjects:
            if attr is not None and getattr(subject, attr, None) is None:
                continue
            yield subject

    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of dataset metadata, one row per subject.
        'channel_names' is excluded.

        Returns
        -------
        pandas.DataFrame
            'Group', 'Subject ID', then the renamed metadata columns.
            Empty (with columns) if no subject has dataset metadata.
        """
        rows = [
            {
                "Group": subject.subject_label,
                "Subject ID": subject.subject_id,
                **_dataset_row(subject.dataset),
            }
            for subject in self._iter_subjects("dataset")
        ]
        # Building rows directly avoids pd.concat of one-row frames, which
        # previously left every row with index 0 and raised on an empty group.
        return pd.DataFrame(rows) if rows else \
            pd.DataFrame(columns=DATASET_ID_COLUMNS_GROUP)

    def get_avalanche_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of avalanche statistics, one row per
        subject x condition.

        Sizes and durations are pooled (concatenated) across matching trials;
        flexibility and avalanche count are averaged.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Group', 'Subject ID', 'Condition', 'Aval Sizes',
            'Aval Durations', 'Mean Flexibility', 'Mean Count'.
        """
        rows = []
        for subject in self._iter_subjects("avalanche_results"):
            by_cond = _group_trials_by_condition(subject.avalanche_results)

            for cond, trials in by_cond.items():
                sizes, durations = [], []
                for trial in trials:
                    sizes.extend(trial.sizes)  # pooled, stays flat
                    durations.extend(trial.durations)

                rows.append({
                    "Group": subject.subject_label,
                    "Subject ID": subject.subject_id,
                    "Condition": _na(cond),
                    "Aval Sizes": sizes,
                    "Aval Durations": durations,
                    "Mean Flexibility": float(np.mean([t.flexibility for t in trials])),
                    "Mean Count": float(np.mean([t.avalanche_count for t in trials])),
                })

        return pd.DataFrame(rows, columns=AVALANCHE_COLUMNS_GROUP)

    def get_patterns_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of avalanche patterns, one row per trial.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Group', 'Subject ID', 'Condition', 'Trial ID',
            'Patterns', 'Temporal Patterns'.
        """
        rows = [
            {
                "Group": subject.subject_label,
                "Subject ID": subject.subject_id,
                "Condition": _na(trial.trial_label),
                "Trial ID": trial.trial_id,
                "Patterns": trial.patterns,
                "Temporal Patterns": trial.patterns_temporal,
            }
            for subject in self._iter_subjects("avalanche_results")
            for trial in subject.avalanche_results.trials
        ]
        return pd.DataFrame(rows, columns=PATTERNS_COLUMNS_GROUP)

    def get_atm_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of mean ATMs, one row per subject x condition.
        ATMs of matching trials are averaged element-wise.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Group', 'Subject ID', 'Condition', 'ATM Method',
            'Mean ATM'. Each 'Mean ATM' cell holds a 2-D numpy array.
        """
        rows = []
        for subject in self._iter_subjects("atm_results"):
            by_cond = _group_trials_by_condition(subject.atm_results)

            for cond, trials in by_cond.items():
                methods = {t.method for t in trials}
                rows.append({
                    "Group": subject.subject_label,
                    "Subject ID": subject.subject_id,
                    "Condition": _na(cond),
                    "ATM Method": methods.pop() if len(methods) == 1
                    else "/".join(sorted(methods)),
                    "Mean ATM": np.mean([t.mean_atm for t in trials], axis=0),
                })

        return pd.DataFrame(rows, columns=ATM_COLUMNS_GROUP)

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of network metrics, one row per
        subject x condition.

        Metrics are averaged across matching trials: element-wise for
        array-valued metrics, plain mean for scalars. Metric columns are
        detected dynamically from the MetricsObj attributes.

        Returns
        -------
        pandas.DataFrame
            Base columns 'Group', 'Subject ID', 'Condition', 'ATM Method'
            plus one column per metric.
        """
        rows = []
        for subject in self._iter_subjects("metrics_results"):
            by_cond = _group_trials_by_condition(subject.metrics_results)

            for cond, trials in by_cond.items():
                # Collect every metric across this condition's trials.
                # dict preserves first-seen order, so column order is stable.
                collected = {}
                for trial in trials:
                    for column, value in _metric_items(trial):
                        collected.setdefault(column, []).append(value)

                methods = {t.atm_method for t in trials}
                row = {
                    "Group": subject.subject_label,
                    "Subject ID": subject.subject_id,
                    "Condition": _na(cond),
                    "ATM Method": methods.pop() if len(methods) == 1
                    else "/".join(sorted(methods)),
                }
                row.update({col: _mean_metric(vals)
                            for col, vals in collected.items()})
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=METRICS_BASE_COLUMNS_GROUP)

        df = pd.DataFrame(rows)
        metric_cols = [c for c in df.columns if c not in METRICS_BASE_COLUMNS_GROUP]
        return df[METRICS_BASE_COLUMNS_GROUP + metric_cols]

    def get_averaged_metrics_dataframe(self) -> pd.DataFrame:
        """
        Builds a DataFrame of node-level metrics averaged per
        subject x condition.

        Only array-valued (per-node) metrics listed in NODE_LEVEL_METRICS are
        included; scalar summaries are omitted. Use get_metrics_dataframe()
        for the full set.

        Returns
        -------
        pandas.DataFrame
            Columns: 'Group', 'Subject ID', 'Condition', plus one column per
            node-level metric, each holding a 1-D array of per-node means.
        """
        base_columns = ["Group", "Subject ID", "Condition"]

        rows = []
        for subject in self._iter_subjects("metrics_results"):
            by_cond = _group_trials_by_condition(subject.metrics_results)

            for cond, trials in by_cond.items():
                row = {
                    "Group": subject.subject_label,
                    "Subject ID": subject.subject_id,
                    "Condition": _na(cond),
                }
                for attr in NODE_LEVEL_METRICS:
                    values = [getattr(t, attr) for t in trials
                              if getattr(t, attr, None) is not None]
                    if values:
                        column = attr.replace("_", " ").title()
                        row[column] = np.mean(np.asarray(values), axis=0)
                rows.append(row)

        if not rows:
            return pd.DataFrame(columns=base_columns)

        df = pd.DataFrame(rows)
        metric_cols = [c for c in df.columns if c not in base_columns]
        return df[base_columns + metric_cols]

    def get_summary_dataframe(self):

        df_list = []

        for subj in self.subjects:
            df_subj = subj.get_summary_dataframe()
            df_subj.insert(0, 'Group', subj.subject_label)
            df_list.append(df_subj)

        df_group = pd.concat(df_list)
        return df_group

    def save(self, out_path, prefix):
        """
        Save all available group-level results to disk.

        Depending on which analyses have been performed for every subject,
        this method exports the corresponding summary DataFrames as CSV files.
        A serialised copy of the entire ``GroupResults`` object is always saved
        as a pickle file.

        The following files may be created inside `out_path`:

        - ``<prefix>_avalanche_results.csv`` – Avalanche statistics
          (sizes, durations, flexibility, count) per subject and condition.
        - ``<prefix>_avalanche_patterns_results.csv`` – Spatial and temporal
          avalanche patterns per trial.
        - ``<prefix>_atm_results.csv`` – Mean Avalanche Transition Matrices
          per subject and condition.
        - ``<prefix>_metrics_results.csv`` – Graph-theory / network metrics
          per subject and condition.
        - ``<prefix>_group_results.pkl`` – Pickle of the full
          ``GroupResults`` instance (always written).

        Parameters
        ----------
        out_path : str
            Path to the output directory. The directory must already exist.
        prefix : str
            String prepended to every output file name, useful for
            distinguishing different analysis runs or parameter sets.
        """
        #if self.all_subjects_have("avalanche_results"):
        #    df_aval = self.get_avalanche_dataframe()
        #    df_aval.to_csv(os.path.join(out_path,f"{prefix}_avalanche_results.csv"), index=False)

        #    df_pattern = self.get_patterns_dataframe()
        #    df_pattern.to_csv(os.path.join(out_path,f"{prefix}_avalanche_patterns_results.csv"), index=False)

        #if self.all_subjects_have("atm_results"):
        #    df_atm = self.get_atm_dataframe()
        #    df_atm.to_csv(os.path.join(out_path, f"{prefix}_atm_results.csv"), index=False)

        #if self.all_subjects_have("metrics_results"):
        #    df_metrics = self.get_metrics_dataframe()
        #    df_metrics.to_csv(os.path.join(out_path, f"{prefix}_metrics_results.csv"), index=False)

        #report = generate_report(results=self, out_path=out_path, prefix=prefix)
        #report.save(os.path.join(out_path, f"{prefix}_metrics_results.csv"))

        df_group = self.get_summary_dataframe()
        df_group.to_csv(os.path.join(out_path,f"{prefix}_group_results.csv"), index=False)

        with open(os.path.join(out_path,f"{prefix}_group_results.pkl"),'wb') as f:
            pickle.dump(self, f)

