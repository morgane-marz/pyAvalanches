import pickle
import numpy as np
import pytest
import mne
import pandas as pd
from unittest.mock import MagicMock, patch

from pyAvalanches.subject_group_results import Subject
from pyAvalanches.avalanches import AvalancheObj, TrialAvalanches
from pyAvalanches.transition_matrices import ATMObj, TrialATM
from pyAvalanches.connectivity_metrics import MetricsObj, TrialMetrics
from pyAvalanches.dataset import Dataset

MODULE = 'pyAvalanches.dataset'

def avalanche_matrix(n_regions=3, duration=4):
    """
    A small binarized avalanche matrix, shape (n_regions, duration).

    Deterministic pattern: region i is active from time i onwards, so every
    region is active at least once and the matrix is never all-zeros.
    """
    mat = np.zeros((n_regions, duration), dtype=int)
    for i in range(n_regions):
        mat[i, min(i, duration - 1):] = 1
    return mat

def symmetric_atm(n_regions=3):
    """
    A weighted symmetric ATM with zero diagonal.
    Valid input for bct / networkx: fully connected, so path-based metrics
    do not hit infinities.
    """
    m = np.array([[0.0, 0.5, 0.2],
                  [0.5, 0.0, 0.8],
                  [0.2, 0.8, 0.0]])
    if n_regions == 3:
        return m
    # generic fallback for other sizes
    rng = np.random.default_rng(0)
    a = rng.random((n_regions, n_regions))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 0.0)
    return a


def asymmetric_atm(n_regions=3):
    """A directed (non-symmetric) ATM, as produced by the 'ws' method."""
    if n_regions == 3:
        return np.array([[0.0, 0.7, 0.3],
                         [0.1, 0.0, 0.9],
                         [0.4, 0.2, 0.0]])
    rng = np.random.default_rng(1)
    a = rng.random((n_regions, n_regions))
    np.fill_diagonal(a, 0.0)
    return a

def make_avalanche_obj(trial_id=0, n_avalanches=2, n_regions=3, label=None):
    """
    A populated AvalancheObj.

    durations and sizes are derived from the matrices so the object is
    internally consistent (a test asserting size == sum(matrix) will pass).
    """
    matrices  = [avalanche_matrix(n_regions, duration=d)
                 for d in range(2, 2 + n_avalanches)]
    durations = [m.shape[1] for m in matrices]
    sizes     = [int(m.sum()) for m in matrices]
    patterns  = [np.where(m.sum(axis=1) > 0, 1, 0) for m in matrices]

    return AvalancheObj(
        trial_id=trial_id,
        avalanches_mat=matrices,
        time_bins=1,
        sizes=sizes,
        durations=durations,
        patterns=patterns,
        patterns_temporal=[np.argmax(m, axis=1) + 1 for m in matrices],
        bounds=[(0, d - 1) for d in durations],
        flexibility=len(np.unique(patterns, axis=0)),
        avalanche_count=n_avalanches,
        trial_label=label,
    )

def make_atm_obj(trial_id=0, symmetric=True, method="standard",
                 label=None, n_regions=3, n_avalanches=2, mean_atm=None):
    """
    A populated ATMObj.

    symmetric and method are kept consistent by default:
      symmetric=True  → method='standard'
      symmetric=False → caller usually wants method='ws'
    trial_label is set after construction, mirroring ATMPipeline.process().
    """
    if mean_atm is None:
        mean_atm = symmetric_atm(n_regions) if symmetric else asymmetric_atm(n_regions)

    obj = ATMObj(
        trial_id=trial_id,
        avalanche_transition_matrices=[mean_atm.copy() for _ in range(n_avalanches)],
        method=method,
        symmetric=symmetric,
        mean_atm=mean_atm,
    )
    obj.trial_label = label
    return obj

def make_metrics_obj(trial_id=0, atm_method="standard", label=None, n_regions=3):
    """
    A populated MetricsObj, with the attributes that NetworkMetricsPipeline
    sets for the symmetric metric set.
    """
    obj = MetricsObj(trial_id=trial_id, atm_method=atm_method, trial_label=label)

    obj.node_strength                  = np.linspace(0.5, 1.5, n_regions)
    obj.avg_node_strength              = float(np.mean(obj.node_strength))
    obj.clustering_coefficient         = np.full(n_regions, 0.4)
    obj.avg_clustering_coefficient     = 0.4
    obj.eigenvector_centrality         = np.full(n_regions, 1 / np.sqrt(n_regions))
    obj.avg_eigenvector_centrality     = float(1 / np.sqrt(n_regions))
    obj.modularity                     = 0.25
    obj.char_path                      = 1.75
    obj.mean_mat                       = 0.3
    obj.var_mat                        = 0.08
    obj.var_coeff                      = 0.94
    obj.skew_mat                       = np.zeros(n_regions)
    return obj

DEFAULT_TRIAL_IDS    = [0, 1, 2]
DEFAULT_TRIAL_LABELS = ["rest", "task", "rest"]

def make_trial_avalanches(trial_ids=None, labels=None, n_regions=3):
    """A TrialAvalanches holding one AvalancheObj per requested trial."""
    trial_ids = DEFAULT_TRIAL_IDS    if trial_ids is None else trial_ids
    labels    = DEFAULT_TRIAL_LABELS if labels    is None else labels

    return TrialAvalanches([
        make_avalanche_obj(trial_id=tid, label=lab, n_regions=n_regions)
        for tid, lab in zip(trial_ids, labels)
    ])

def make_trial_atm(trial_ids=None, labels=None, symmetric=True,
                   method="standard", n_regions=3):
    """A TrialATM holding one ATMObj per requested trial."""
    trial_ids = DEFAULT_TRIAL_IDS    if trial_ids is None else trial_ids
    labels    = DEFAULT_TRIAL_LABELS if labels    is None else labels

    return TrialATM([
        make_atm_obj(trial_id=tid, label=lab, symmetric=symmetric,
                     method=method, n_regions=n_regions)
        for tid, lab in zip(trial_ids, labels)
    ])


def make_trial_metrics(trial_ids=None, labels=None,
                       atm_method="standard", n_regions=3):
    """A TrialMetrics holding one MetricsObj per requested trial."""
    trial_ids = DEFAULT_TRIAL_IDS    if trial_ids is None else trial_ids
    labels    = DEFAULT_TRIAL_LABELS if labels    is None else labels

    return TrialMetrics([
        make_metrics_obj(trial_id=tid, label=lab,
                         atm_method=atm_method, n_regions=n_regions)
        for tid, lab in zip(trial_ids, labels)
    ])

def make_data_dict(n_trials=3, n_channels=3, n_times=500, trial_ids=None, labels=None):
    """
    A valid info dict, matching what AvalanchePipeline.process() expects.
    Note: 'trial_ids' is included, since it is an optional key there.
    """
    np.random.seed(42)
    return {
        "data_array":    np.random.rand(n_trials, n_channels, n_times),
        "samp_freq":     250.0,
        "fmin":          0.5,
        "fmax":          40.0,
        "channel_names": [f"ch{i}" for i in range(n_channels)],
        "trial_labels":  DEFAULT_TRIAL_LABELS[:n_trials] if labels is None else labels,
        "trial_ids":     list(range(n_trials)) if trial_ids is None else trial_ids,
        "t_max":         2.0,
    }

def make_mock_raw(n_channels=3, n_times=500, samp_freq=250.0):
    """
    A stand-in for mne.io.BaseRaw.

    spec= is essential: it makes isinstance(mock, mne.io.BaseRaw) return True,
    so code guarded by `if isinstance(data, mne.io.BaseRaw)` takes the right
    branch.  Requires mne to be importable.
    """
    raw = MagicMock(spec=mne.io.BaseRaw)
    raw.ch_names = [f"ch{i}" for i in range(n_channels)]
    raw.info = {"sfreq": samp_freq,
                "ch_names": [f"ch{i}" for i in range(n_channels)],
                "lowpass": 35,
                "highpass": 8
                }
    raw.get_data.return_value = np.random.default_rng(0).standard_normal(
        (n_channels, n_times)
    )
    return raw

def make_mock_epochs(n_epochs=3, n_channels=3, n_times=100, samp_freq=250.0):
    """A stand-in for mne.BaseEpochs, same spec= reasoning as above."""
    epochs = MagicMock(spec=mne.Epochs)
    epochs.ch_names = [f"ch{i}" for i in range(n_channels)]
    epochs.info = {"sfreq": samp_freq,
                   "ch_names": [f"ch{i}" for i in range(n_channels)],
                   "lowpass":  40,
                   "highpass": 1
                   }
    epochs.get_data.return_value = np.random.default_rng(1).standard_normal(
        (n_epochs, n_channels, n_times)
    )
    epochs.events = np.column_stack([
        np.arange(n_epochs) * n_times,
        np.zeros(n_epochs, dtype=int),
        np.ones(n_epochs, dtype=int),
    ])
    return epochs

def make_subject(subject_id="sub-01", subject_label=None,
                 with_avalanches=True, with_atm=True, with_metrics=True,
                 with_dataset=False,
                 trial_ids=None, labels=None,
                 symmetric=True, method="standard", n_regions=3):
    """
    Builds a Subject with any combination of results present or absent.

    All result containers share the same trial_ids and labels, so tests can
    check that IDs line up across the three analysis stages.

    Parameters
    ----------
    with_avalanches, with_atm, with_metrics : bool
        Include that result container, or leave it as None.
    with_dataset : bool
        If True, attach an info dict as the dataset.
    symmetric, method : bool, str
        Passed through to the ATM and metrics builders, so a
        'ws' / non-symmetric Subject can be built in one call.
    """
    return Subject(
        subject_id=subject_id,
        subject_label=subject_label,
        avalanche_results=(
            make_trial_avalanches(trial_ids, labels, n_regions)
            if with_avalanches else None
        ),
        atm_results=(
            make_trial_atm(trial_ids, labels, symmetric, method, n_regions)
            if with_atm else None
        ),
        metrics_results=(
            make_trial_metrics(trial_ids, labels, method, n_regions)
            if with_metrics else None
        ),
        dataset=make_data_dict() if with_dataset else None,
    )

@pytest.fixture
def minimal_subject():
    """Only subject_id set; every optional field is None."""
    return Subject(subject_id="sub-01")

@pytest.fixture
def full_subject():
    """All three result stages populated, symmetric / 'standard'."""
    return make_subject(subject_id="sub-01", subject_label="patient")

class TestSubjectInit:

    def test_subject_id_is_required(self):
        """subject_id has no default, so omitting it must fail."""
        with pytest.raises(TypeError):
            Subject()

    @pytest.mark.parametrize("subject_id", [1, 42, "sub-01", "P07"])
    def test_subject_id_accepts_int_and_str(self, subject_id):
        assert Subject(subject_id=subject_id).subject_id == subject_id

    def test_optional_fields_default_to_none(self, minimal_subject):
        for attr in ["subject_label", "avalanche_results",
                     "atm_results", "metrics_results", "dataset"]:
            assert getattr(minimal_subject, attr) is None

    def test_all_fields_stored(self, full_subject):
        assert full_subject.subject_id == "sub-01"
        assert full_subject.subject_label == "patient"
        assert full_subject.avalanche_results is not None
        assert full_subject.atm_results is not None
        assert full_subject.metrics_results is not None

    def test_trial_ids_match_across_all_stages(self, full_subject):
        """The same trials must be identifiable at every stage."""
        aval_ids = full_subject.avalanche_results.trial_indexes
        atm_ids = [t.trial_id for t in full_subject.atm_results.trials]
        metrics_ids = [t.trial_id for t in full_subject.metrics_results.trials]

        assert aval_ids == atm_ids == metrics_ids

    def test_labels_match_across_all_stages(self, full_subject):
        aval_labels = full_subject.avalanche_results.trial_labels
        atm_labels = [t.trial_label for t in full_subject.atm_results.trials]
        metrics_labels = [t.trial_label for t in full_subject.metrics_results.trials]

        assert aval_labels == atm_labels == metrics_labels

    @pytest.mark.parametrize("bad_dataset", [
        [1, 2, 3],
        (1, 2, 3),
        "a_file_path.fif",
        42,
        np.zeros((3, 3, 3)),  # a bare array is NOT accepted – must be a dict
    ])
    def test_unsupported_type_raises_type_error(self, bad_dataset):
        with pytest.raises(TypeError, match="must be Raw or Epochs object or dict"):
            Subject(subject_id="sub-01", dataset=bad_dataset)

    def test_dict_dataset_is_stored(self):
        ds = make_data_dict()
        subject = Subject(subject_id='sub_01', dataset=ds)
        assert isinstance(subject.dataset, dict)
        assert "samp_freq" in subject.dataset

    def test_raw_dataset_is_stored(self):
        ds = make_mock_raw()
        s = Subject(subject_id='sub_01', dataset=ds)
        assert isinstance(s.dataset, dict)
        #assert "samp_freq" in subject.dataset
        assert s.dataset["n_channels"] == 3
        assert s.dataset["n_time_points"] == 500
        assert s.dataset["n_trials"] == 1
        assert s.dataset["samp_freq"] == 250.0
        assert s.dataset["fmin"] == 8
        assert s.dataset["fmax"] == 35

    def test_epoch_dataset_is_stored(self):
        ds = make_mock_epochs()
        s = Subject(subject_id='sub_01', dataset=ds)
        assert isinstance(s.dataset, dict)
        assert s.dataset["n_channels"] == 3
        assert s.dataset["n_time_points"] == 100
        assert s.dataset["n_trials"] == 3
        assert s.dataset["samp_freq"] == 250.0
        assert s.dataset["fmin"] == 1
        assert s.dataset["fmax"] == 40

    def test_data_from_dict_is_not_stored(self):
        ds = make_data_dict()
        subject = Subject(subject_id='sub_01', dataset=ds)
        assert "data_array" not in subject.dataset

    def test_expected_keys_present(self):
        ds = make_data_dict()
        s = Subject(subject_id="sub-01", dataset=ds)

        expected = {"n_channels", "n_time_points", "n_trials", "conditions",
                    "channel_names", "samp_freq", "fmax", "fmin"}
        assert set(s.dataset) == expected

    def test_shape_derived_correctly(self):
        """
        data_array shape is (n_trials, n_channels, n_time_points) →
        shape[0]=trials, shape[1]=channels, shape[2]=time points.
        Distinct numbers catch an axis mix-up.
        """
        ds = make_data_dict()
        s = Subject(subject_id="sub-01", dataset=ds)

        assert s.dataset["n_trials"] == 3
        assert s.dataset["n_channels"] == 3
        assert s.dataset["n_time_points"] == 500

    def test_metadata_copied_verbatim(self):
        dataset_dict = make_data_dict()
        s = Subject(subject_id="sub-01", dataset=dataset_dict)

        assert s.dataset["samp_freq"] == dataset_dict["samp_freq"]
        assert s.dataset["fmin"] == dataset_dict["fmin"]
        assert s.dataset["fmax"] == dataset_dict["fmax"]
        assert s.dataset["channel_names"] == dataset_dict["channel_names"]

    @pytest.mark.parametrize("missing_key", [
        "data_array", "trial_labels", "channel_names",
        "samp_freq", "fmin", "fmax",
    ])
    def test_missing_key_raises_key_error(self, missing_key):

        """
        Every key is accessed directly, so an incomplete dict raises KeyError.
        TODO Documents which keys are mandatory.
        """
        ds = make_data_dict()
        del ds[missing_key]

        with pytest.raises(KeyError):
            Subject(subject_id="sub-01", dataset=ds)

class TestGetMeanAtmCond:

    @pytest.fixture
    def subject_two_conds(self):
        """
        4 trials with hand-chosen matrices:
          trial 0 'rest' → all 1
          trial 1 'MI'   → all 10
          trial 2 'rest' → all 3     → mean of 'rest' = 2
          trial 3 'MI'   → all 20    → mean of 'MI'   = 15
        """
        trials = [
            make_atm_obj(0, label="rest", mean_atm=np.full((2, 2), 1.0)),
            make_atm_obj(1, label="task", mean_atm=np.full((2, 2), 10.0)),
            make_atm_obj(2, label="rest", mean_atm=np.full((2, 2), 3.0)),
            make_atm_obj(3, label="task", mean_atm=np.full((2, 2), 20.0)),
        ]
        return Subject(subject_id="sub-01", atm_results=TrialATM(trials))

    @pytest.mark.parametrize("cond, mean_atm",[
        ("rest", np.full((2, 2), 2.0)),
        ("task", np.full((2, 2), 15.0))
    ])
    def test_averages_only_matching_trials(self, subject_two_conds, cond, mean_atm):
        result = subject_two_conds.get_mean_atm_cond(cond)
        np.testing.assert_allclose(result, mean_atm)

    def test_output_shape_matches_single_atm(self, subject_two_conds):
        assert subject_two_conds.get_mean_atm_cond("rest").shape == (2, 2)

    def test_single_matching_trial_returns_that_matrix(self):
        trials = [
            make_atm_obj(0, label="rest", mean_atm=np.full((2, 2), 7.0)),
            make_atm_obj(1, label="task", mean_atm=np.full((2, 2), 9.0)),
        ]
        s = Subject(subject_id="sub-01", atm_results=TrialATM(trials))

        np.testing.assert_allclose(s.get_mean_atm_cond("task"), np.full((2, 2), 9.0))
        np.testing.assert_allclose(s.get_mean_atm_cond("rest"), np.full((2, 2), 7.0))

    def test_non_contiguous_trial_ids_work(self):
        """The loop uses trial_indexes + get_trial_data, so any IDs are fine."""
        trials = [
            make_atm_obj(5, label="rest", mean_atm=np.full((2, 2), 2.0)),
            make_atm_obj(9, label="rest", mean_atm=np.full((2, 2), 4.0)),
        ]
        s = Subject(subject_id="sub-01", atm_results=TrialATM(trials))

        np.testing.assert_allclose(s.get_mean_atm_cond("rest"), np.full((2, 2), 3.0))

    def test_invalid_method_raises_value_error(self, subject_two_conds):
        with pytest.raises(ValueError, match="no trials with condition"):
            subject_two_conds.get_mean_atm_cond("wrong_condition")

    def test_no_atm_results_raises_value_error(self):
        s = Subject(subject_id="sub-01")
        with pytest.raises(ValueError, match="is not available"):
            s.get_mean_atm_cond("rest")

class TestGetAvalancheDataframe:

    def test_returns_dataframe(self, full_subject):
        assert isinstance(full_subject.get_avalanche_dataframe(), pd.DataFrame)

    def test_one_row_per_trial(self, full_subject):
        df = full_subject.get_avalanche_dataframe()
        assert len(df) == full_subject.avalanche_results.trial_count

    def test_expected_columns(self, full_subject):
        df = full_subject.get_avalanche_dataframe()

        expected = ["Subject ID", "Trial ID", "Condition", "Aval Count",
                    "Aval Sizes", "Aval Durations", "Flexibility",
                    "Patterns", "Temporal Patterns"]
        assert list(df.columns) == expected

    def test_subject_id_on_every_row(self, full_subject):
        df = full_subject.get_avalanche_dataframe()
        assert (df["Subject ID"] == "sub-01").all()

    def test_all_trial_ids_present(self, full_subject):
        """Compared as a set, since row order follows arbitrary set iteration."""
        df = full_subject.get_avalanche_dataframe()
        assert set(df["Trial ID"]) == set(full_subject.avalanche_results.trial_indexes)

    def test_condition_paired_with_correct_trial(self, full_subject):
        """
        Labels are rest/task/rest for trials 0/1/2.  Sorting by Trial ID makes
        this check order-independent.
        """
        df = full_subject.get_avalanche_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)
        assert list(df["Condition"]) == ["rest", "task", "rest"]

    def test_values_match_source_objects(self, full_subject):
        df = full_subject.get_avalanche_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)

        for _, row in df.iterrows():
            trial = full_subject.avalanche_results.get_trial_data(row["Trial ID"])
            assert row["Aval Count"] == trial.avalanche_count
            assert row["Aval Sizes"] == trial.sizes
            assert row["Aval Durations"] == trial.durations
            assert row["Flexibility"] == trial.flexibility

    def test_non_contiguous_trial_ids_preserved(self):
        s = Subject(subject_id="sub-01",
                    avalanche_results= make_trial_avalanches(trial_ids=[5, 7, 9]))

        df = s.get_avalanche_dataframe()

        assert set(df["Trial ID"]) == {5, 7, 9}

    def test_none_labels_become_pd_na(self):
        """
        set([None, None, None]) == {None}, and `trial_label == None` is True,
        so rows ARE produced and Condition is filled with pd.NA.
        """
        s = Subject(subject_id="sub-01",
                    avalanche_results=make_trial_avalanches(
                        labels=[None, None, None]))

        df = s.get_avalanche_dataframe()

        assert len(df) == 3
        assert df["Condition"].isna().all()

    def test_no_avalanche_results_raises_value_error(self):
        with pytest.raises(ValueError, match="is not available"):
            Subject(subject_id="sub-01").get_avalanche_dataframe()

class TestGetPatternsDataframe:

    def test_expected_columns(self, full_subject):
        df = full_subject.get_patterns_dataframe()

        expected = ["Subject ID", "Trial ID", "Condition",
                    "Patterns", "Temporal Patterns"]
        assert list(df.columns) == expected

    def test_one_row_per_trial(self, full_subject):
        df = full_subject.get_patterns_dataframe()
        assert len(df) == full_subject.avalanche_results.trial_count

    def test_patterns_match_source_objects(self, full_subject):
        df = full_subject.get_patterns_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)

        for _, row in df.iterrows():
            trial = full_subject.avalanche_results.get_trial_data(row["Trial ID"])
            for actual, expected in zip(row["Patterns"], trial.patterns):
                np.testing.assert_array_equal(actual, expected)

    def test_no_avalanche_results_raises_value_error(self):
        with pytest.raises(ValueError, match="is not available"):
            Subject(subject_id="sub-01").get_patterns_dataframe()

class TestGetAtmDataframe:

    def test_expected_columns(self, full_subject):
        df = full_subject.get_atm_dataframe()
        assert list(df.columns) == ["Subject ID", "Trial ID", "Condition", "ATM Method", "Mean ATM"]

    def test_one_row_per_trial(self, full_subject):
        df = full_subject.get_atm_dataframe()
        assert len(df) == len(full_subject.atm_results.trials)

    def test_mean_atm_matches_source(self, full_subject):
        df = full_subject.get_atm_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)

        for _, row in df.iterrows():
            trial = full_subject.atm_results.get_trial_data(row["Trial ID"])
            np.testing.assert_array_equal(row["Mean ATM"], trial.mean_atm)

    def test_mean_atm_cells_hold_arrays(self, full_subject):
        """Each cell is a full 2-D matrix, not a flattened or scalar value."""
        df = full_subject.get_atm_dataframe()

        assert isinstance(df["Mean ATM"].iloc[0], np.ndarray)
        assert df["Mean ATM"].iloc[0].ndim == 2

    def test_no_atm_results_raises_value_error(self):
        with pytest.raises(ValueError, match="is not available"):
            Subject(subject_id="sub-01").get_atm_dataframe()

class TestGetMetricsDataframe:

    def test_base_columns_present(self, full_subject):
        df = full_subject.get_metrics_dataframe()

        for col in ["Subject ID", "Trial ID", "Condition", "ATM Method"]:
            assert col in df.columns

    def test_base_columns_come_first(self, full_subject):
        df = full_subject.get_metrics_dataframe()

        assert list(df.columns)[:4] == ["Subject ID", "Trial ID",
                                        "Condition", "ATM Method"]

    def test_one_row_per_trial(self, full_subject):
        df = full_subject.get_metrics_dataframe()
        assert len(df) == len(full_subject.metrics_results.trials)

    @pytest.mark.parametrize("attr, column", [
        ("node_strength",              "Node Strength"),
        ("avg_node_strength",          "Avg Node Strength"),
        ("clustering_coefficient",     "Clustering Coefficient"),
        ("avg_clustering_coefficient", "Avg Clustering Coefficient"),
        ("eigenvector_centrality",     "Eigenvector Centrality"),
        ("modularity",                 "Modularity"),
        ("char_path",                  "Char Path"),
        ("mean_mat",                   "Mean Mat"),
        ("var_coeff",                  "Var Coeff"),
    ])
    def test_snake_case_attributes_become_title_case_columns(
        self, full_subject, attr, column
    ):
        """attr.replace('_', ' ').title() – verified per attribute."""
        df = full_subject.get_metrics_dataframe()
        assert column in df.columns

    def test_excluded_attributes_absent(self, full_subject):
        """
        trial_id / trial_label / atm_method are handled by the base columns,
        so their auto-generated versions must not appear.
        """
        df = full_subject.get_metrics_dataframe()

        for col in ["Trial Id", "Trial Label", "Atm Method"]:
            assert col not in df.columns

    def test_only_computed_metrics_become_columns(self):
        """A MetricsObj with a single metric yields a single metric column."""
        obj = MetricsObj(trial_id=0, atm_method="standard", trial_label="rest")
        obj.modularity = 0.5

        s = Subject(subject_id="sub-01", metrics_results=TrialMetrics([obj]))
        df = s.get_metrics_dataframe()

        assert "Modularity" in df.columns
        assert "Node Strength" not in df.columns

    def test_metric_values_match_source(self, full_subject):
        df = full_subject.get_metrics_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)

        for _, row in df.iterrows():
            trial = full_subject.metrics_results.get_trial_data(row["Trial ID"])
            assert row["Modularity"] == trial.modularity
            np.testing.assert_array_equal(row["Node Strength"], trial.node_strength)

    def test_mismatched_metrics_across_trials_give_nan(self):
        """
        Trials with different attribute sets are unioned into one frame, so
        the missing cells become NaN rather than raising.
        """
        a = MetricsObj(trial_id=0, atm_method="standard", trial_label="rest")
        a.modularity = 0.5
        b = MetricsObj(trial_id=1, atm_method="standard", trial_label="rest")
        b.char_path = 2.0

        s = Subject(subject_id="sub-01", metrics_results=TrialMetrics([a, b]))
        df = s.get_metrics_dataframe()
        df = df.sort_values("Trial ID").reset_index(drop=True)

        assert df.loc[0, "Modularity"] == 0.5
        assert pd.isna(df.loc[0, "Char Path"])
        assert pd.isna(df.loc[1, "Modularity"])

    def test_no_metrics_results_raises_value_error(self):
        with pytest.raises(ValueError, match="is not available"):
            Subject(subject_id="sub-01").get_metrics_dataframe()

class TestGetDatasetDataframe:

    @pytest.fixture
    def subject_with_dataset(self):
        dataset_dict = make_data_dict()
        return Subject(subject_id="sub-01", dataset=dataset_dict)

    def test_returns_single_row(self, subject_with_dataset):
        """Dataset metadata describes the whole recording, so one row."""
        assert len(subject_with_dataset.get_dataset_dataframe()) == 1

    def test_channel_names_excluded(self, subject_with_dataset):
        """channel_names is a long list, deliberately dropped from the report."""
        df = subject_with_dataset.get_dataset_dataframe()
        assert "channel_names" not in df.columns
        assert "Channel Names" not in df.columns

    def test_subject_id_is_first_column(self, subject_with_dataset):
        df = subject_with_dataset.get_dataset_dataframe()
        assert list(df.columns)[0] == "Subject ID"

    @pytest.mark.parametrize("renamed", [
        "Highpass", "Lowpass", "Number of Channels/ROIs", "Time Points",
        "Number of Trials", "Sampling Frequency", "Trial Conditions",
    ])
    def test_columns_are_renamed(self, subject_with_dataset, renamed):
        df = subject_with_dataset.get_dataset_dataframe()
        assert renamed in df.columns

    def test_no_raw_keys_remain(self, subject_with_dataset):
        """Every raw key must have been renamed."""
        df = subject_with_dataset.get_dataset_dataframe()

        for raw_key in ["fmin", "fmax", "n_channels", "n_time_points",
                        "n_trials", "samp_freq", "conditions"]:
            assert raw_key not in df.columns

    def test_values_match_dataset_info(self):
        ds = make_data_dict()
        s = Subject(subject_id="sub-01", dataset=ds)

        df = s.get_dataset_dataframe()

        assert df["Number of Trials"].iloc[0]        == 3
        assert df["Number of Channels/ROIs"].iloc[0] == 3
        assert df["Time Points"].iloc[0]             == 500
        assert df["Highpass"].iloc[0]                == 0.5
        assert df["Lowpass"].iloc[0]                 == 40.0

    def test_conditions_cell_holds_a_set(self, subject_with_dataset):
        df = subject_with_dataset.get_dataset_dataframe()
        assert df["Trial Conditions"].iloc[0] == {"rest", "task"}

    def test_no_dataset_raises_value_error(self):
        with pytest.raises(ValueError, match="is not available"):
            Subject(subject_id="sub-01").get_dataset_dataframe()


class TestSave:

    #def test_all_csv_files_created(self, full_subject, tmp_path):
    #    full_subject.save(str(tmp_path), prefix="sub-01")

    #    for name in ["sub-01_avalanche_results.csv",
    #                 "sub-01_avalanche_patterns_results.csv",
    #                 "sub-01_atm_results.csv",
    #                 "sub-01_metrics_results.csv"]:
    #        assert (tmp_path / name).exists(), f"missing {name}"

    def test_pickle_file_created(self, full_subject, tmp_path):
        full_subject.save(str(tmp_path), prefix="sub-01")
        assert (tmp_path / "sub-01_subject_results.pkl").exists()

    #def test_avalanche_files_skipped_when_results_absent(self, tmp_path):
    #    s = make_subject(with_avalanches=False)

    #    s.save(str(tmp_path), prefix="sub-01")

    #    assert not (tmp_path / "sub-01_avalanche_results.csv").exists()
    #    assert not (tmp_path / "sub-01_avalanche_patterns_results.csv").exists()
    #    assert (tmp_path / "sub-01_atm_results.csv").exists()

    #def test_atm_file_skipped_when_results_absent(self, tmp_path):
    #    s = make_subject(with_atm=False)

    #    s.save(str(tmp_path), prefix="sub-01")

    #    assert not (tmp_path / "sub-01_atm_results.csv").exists()

    #def test_metrics_file_skipped_when_results_absent(self, tmp_path):
    #    s = make_subject(with_metrics=False)

    #    s.save(str(tmp_path), prefix="sub-01")

    #    assert not (tmp_path / "sub-01_metrics_results.csv").exists()

    def test_pickle_always_written_even_with_no_results(self, tmp_path):
        Subject(subject_id="sub-01").save(str(tmp_path), prefix="sub-01")
        assert (tmp_path / "sub-01_subject_results.pkl").exists()

    def test_pickle_roundtrip_preserves_subject_id(self, full_subject, tmp_path):
        full_subject.save(str(tmp_path), prefix="sub-01")

        with open(tmp_path / "sub-01_subject_results.pkl", "rb") as f:
            loaded = pickle.load(f)

        assert loaded.subject_id == full_subject.subject_id

    def test_pickle_roundtrip_preserves_results(self, full_subject, tmp_path):
        full_subject.save(str(tmp_path), prefix="sub-01")

        with open(tmp_path / "sub-01_subject_results.pkl", "rb") as f:
            loaded = pickle.load(f)

        assert loaded.avalanche_results.trial_indexes == \
               full_subject.avalanche_results.trial_indexes
        np.testing.assert_array_equal(
            loaded.atm_results.trials[0].mean_atm,
            full_subject.atm_results.trials[0].mean_atm,
        )

    def test_csv_is_readable_and_has_the_right_rows(self, full_subject, tmp_path):
        full_subject.save(str(tmp_path), prefix="sub-01")

        df = pd.read_csv(tmp_path / "sub-01_subject_results.csv")

        assert len(df) == full_subject.avalanche_results.trial_count
        assert "Trial ID" in df.columns

    def test_csv_written_without_index_column(self, full_subject, tmp_path):
        full_subject.save(str(tmp_path), prefix="sub-01")

        df = pd.read_csv(tmp_path / "sub-01_subject_results.csv")

        assert not any(c.startswith("Unnamed") for c in df.columns)

    @pytest.mark.parametrize("prefix", ["sub-01", "run_A", "2024_session1"])
    def test_prefix_used_in_every_filename(self, full_subject, tmp_path, prefix):
        full_subject.save(str(tmp_path), prefix=prefix)

        #assert (tmp_path / f"{prefix}_avalanche_results.csv").exists()
        assert (tmp_path / f"{prefix}_subject_results.pkl").exists()
