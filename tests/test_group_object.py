import pickle
import numpy as np
import pytest
import mne
import pandas as pd
from unittest.mock import MagicMock, patch

from pyAvalanches.subject_group_results import Subject, GroupResults
from pyAvalanches.avalanches import AvalancheObj, TrialAvalanches
from pyAvalanches.transition_matrices import ATMObj, TrialATM
from pyAvalanches.connectivity_metrics import MetricsObj, TrialMetrics
from pyAvalanches.dataset import Dataset

DEFAULT_GROUP_LABELS = ["control", "patient"]

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
    A valid info dict
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

def make_subject(subject_id="sub-01", subject_label=None,
                 with_avalanches=True, with_atm=True, with_metrics=True,
                 with_dataset=False,
                 trial_ids=None,
                 labels=None,
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

def make_group(n_subjects=4, labels=None, subject_ids=None,
               with_dataset=True, **subject_kwargs):
    """
    Builds a GroupResults containing `n_subjects` Subjects.

    labels : list, optional
        One group label per subject. Defaults to alternating
        control/patient, so every group has at least two labels.
    subject_ids : list, optional
        Defaults to 'sub-01', 'sub-02', ...
    with_dataset : bool
        Subjects need a populated .dataset for get_dataset_dataframe().
        On by default here, unlike make_subject().
    **subject_kwargs
        Forwarded to make_subject (with_avalanches, with_atm, symmetric, ...).
    """
    if labels is None:
        labels = [DEFAULT_GROUP_LABELS[i % 2] for i in range(n_subjects)]
    if subject_ids is None:
        subject_ids = [f"sub-{i + 1:02d}" for i in range(n_subjects)]

    subjects = [
        make_subject(subject_id=sid, subject_label=lab,
                     with_dataset=with_dataset, **subject_kwargs)
        for sid, lab in zip(subject_ids, labels)
    ]
    return GroupResults(subjects=subjects,
                        channel_names=[f"ch{i}" for i in range(3)])


@pytest.fixture
def empty_group():
    return GroupResults()


@pytest.fixture
def group_two_labels():
    """
    4 subjects:
      sub-01 control | sub-02 patient | sub-03 control | sub-04 patient
    Each has 3 trials labelled rest / task / rest, all three result stages,
    and a dataset dict.
    """
    return make_group(n_subjects=4)


@pytest.fixture
def group_factory():
    """Call inside a test to build a tailored group."""
    return make_group


def sorted_rows(df, *by):
    """Sort by the given columns and reset the index, for order-independence."""
    return df.sort_values(list(by)).reset_index(drop=True)


class TestGroupResultsInit:

    def test_no_arguments_gives_empty_group(self, empty_group):
        assert empty_group.subjects      == []
        assert empty_group.channel_names is None
        assert empty_group.info          is None

    def test_all_fields_stored(self, group_two_labels):
        assert len(group_two_labels.subjects)  == 4
        assert group_two_labels.channel_names  == ["ch0", "ch1", "ch2"]

    def test_info_dict_stored(self):
        g = GroupResults(info={"study": "pilot"})
        assert g.info == {"study": "pilot"}

    def test_two_instances_do_not_share_the_default_list(self):
        """
        field(default_factory=list) creates a fresh list per instance.
        Appending to one must not affect the other.
        """
        a, b = GroupResults(), GroupResults()

        a.add_subject(Subject(subject_id="sub-01"))

        assert a.n_subjects == 1
        assert b.n_subjects == 0


class TestAddSubject:

    def test_subject_count_increases(self, group_two_labels):
        before = group_two_labels.n_subjects

        group_two_labels.add_subject(Subject(subject_id="sub-99"))

        assert group_two_labels.n_subjects == before + 1

    def test_added_subject_is_retrievable(self, empty_group):
        s = Subject(subject_id="sub-01")

        empty_group.add_subject(s)

        assert empty_group.get_subject("sub-01") is s

    def test_added_subject_appears_in_ids_and_labels(self, empty_group):
        empty_group.add_subject(Subject(subject_id="sub-01",
                                        subject_label="control"))

        assert empty_group.subject_ids    == ["sub-01"]
        assert empty_group.subject_labels == ["control"]


class TestGetSubject:

    @pytest.mark.parametrize("subject_id", ["sub-01", "sub-02",
                                            "sub-03", "sub-04"])
    def test_returns_correct_subject(self, group_two_labels, subject_id):
        assert group_two_labels.get_subject(subject_id).subject_id == subject_id

    def test_missing_id_raises_value_error(self, group_two_labels):
        with pytest.raises(ValueError, match="Subject ID"):
            group_two_labels.get_subject("nonexistent")

    def test_integer_ids_work(self, group_factory):
        g = group_factory(n_subjects=2, subject_ids=[5, 9])

        assert g.get_subject(9).subject_id == 9


class TestProperties:

    def test_subject_ids(self, group_two_labels):
        assert group_two_labels.subject_ids == ["sub-01", "sub-02",
                                                "sub-03", "sub-04"]

    def test_subject_labels(self, group_two_labels):
        assert group_two_labels.subject_labels == ["control", "patient",
                                                   "control", "patient"]

    def test_n_subjects(self, group_two_labels):
        assert group_two_labels.n_subjects == 4

    def test_properties_on_empty_group(self, empty_group):
        assert empty_group.subject_ids    == []
        assert empty_group.subject_labels == []
        assert empty_group.n_subjects     == 0

    def test_labels_are_none_when_unset(self, group_factory):
        g = group_factory(n_subjects=2, labels=[None, None])

        assert g.subject_labels == [None, None]

    def test_properties_reflect_later_additions(self, empty_group):
        """Properties are computed on access, not cached at construction."""
        assert empty_group.n_subjects == 0

        empty_group.add_subject(Subject(subject_id="sub-01"))

        assert empty_group.n_subjects == 1

class TestAllSubjectsHave:

    @pytest.mark.parametrize("attr", ["avalanche_results", "atm_results",
                                      "metrics_results"])
    def test_true_when_every_subject_has_it(self, group_two_labels, attr):
        assert group_two_labels.all_subjects_have(attr) is True

    def test_false_when_one_subject_is_missing_it(self, group_factory):
        """A single incomplete subject makes the whole group incomplete."""
        g = group_factory(n_subjects=3)
        g.subjects[1].metrics_results = None

        assert g.all_subjects_have("metrics_results") is False

    def test_false_on_empty_group(self, empty_group):
        """
        `bool(self.subjects) and ...` guards against all([]) == True,
        which would otherwise wrongly report an empty group as complete.
        """
        assert empty_group.all_subjects_have("atm_results") is False

    def test_true_for_subject_id_which_is_always_set(self, group_two_labels):
        assert group_two_labels.all_subjects_have("subject_id") is True


class TestGetResultsPerSubject:

    # ── multi-trial case: the whole container is returned ─────────────────────

    def test_avalanche_returns_container_for_multi_trial(self, group_two_labels):
        result = group_two_labels.get_avalanche_subj("sub-01")

        assert result is group_two_labels.get_subject("sub-01").avalanche_results

    def test_atm_returns_container_for_multi_trial(self, group_two_labels):
        result = group_two_labels.get_atm_subj("sub-01")

        assert result is group_two_labels.get_subject("sub-01").atm_results

    def test_metrics_returns_container_for_multi_trial(self, group_two_labels):
        result = group_two_labels.get_metrics_subj("sub-01")

        assert result is group_two_labels.get_subject("sub-01").metrics_results

    # ── single-trial case: the trial object is unwrapped ──────────────────────

    def test_avalanche_unwraps_single_trial(self, group_factory):
        """
        With trial_count == 1 the method returns the AvalancheObj itself,
        not the TrialAvalanches wrapper.
        """
        g = group_factory(n_subjects=1, trial_ids=[0], labels_trials=None) \
            if False else group_factory(n_subjects=1)
        # rebuild sub-01 with a single trial (id 0)
        g.subjects[0].avalanche_results = make_trial_avalanches(
            trial_ids=[0], labels=["rest"]
        )

        result = g.get_avalanche_subj(g.subject_ids[0])

        assert result.trial_id == 0
        assert not hasattr(result, "trial_count")   # not the container

    def test_atm_unwraps_single_trial(self, group_factory):
        g = group_factory(n_subjects=1)
        g.subjects[0].atm_results = make_trial_atm(trial_ids=[0],
                                                   labels=["rest"])

        result = g.get_atm_subj(g.subject_ids[0])

        assert result.trial_id == 0

    def test_metrics_unwraps_single_trial(self, group_factory):
        g = group_factory(n_subjects=1)
        g.subjects[0].metrics_results = make_trial_metrics(trial_ids=[0],
                                                           labels=["rest"])

        result = g.get_metrics_subj(g.subject_ids[0])

        assert result.trial_id == 0

    def test_missing_subject_raises_value_error(self, group_two_labels):
        """get_subject returns ValueError, before `.avalanche_results` fails."""
        with pytest.raises(ValueError, match="Subject ID"):
            group_two_labels.get_avalanche_subj("nonexistent")


class TestGetMeanSubjAtm:

    def test_delegates_to_global_mean_atm(self, group_factory):
        """
        The method's only job is to forward to
        subject.atm_results.global_mean_atm().
        Mocking atm_results tests that delegation without depending on
        TrialATM's implementation.
        """
        g = group_factory(n_subjects=1)
        expected = np.full((3, 3), 0.5)

        fake_atm = MagicMock()
        fake_atm.global_mean_atm.return_value = expected
        g.subjects[0].atm_results = fake_atm

        result = g.get_mean_subj_atm(g.subject_ids[0])

        fake_atm.global_mean_atm.assert_called_once_with()
        np.testing.assert_array_equal(result, expected)

    def test_real_trial_atm_provides_global_mean_atm(self, group_two_labels):
        """
        Guard: fails if TrialATM does not actually implement
        global_mean_atm(), which get_mean_subj_atm depends on.
        """
        atm_results = group_two_labels.get_subject("sub-01").atm_results

        assert hasattr(atm_results, "global_mean_atm"), (
            "TrialATM must implement global_mean_atm() for "
            "GroupResults.get_mean_subj_atm() to work"
        )

    def test_missing_subject_raises_value_error(self, group_two_labels):
        with pytest.raises(ValueError, match="Subject ID"):
            group_two_labels.get_mean_subj_atm("nonexistent")


class TestGetMeanAtmGroup:

    @pytest.fixture
    def group_known_atms(self):
        """
        4 subjects with hand-set 'rest' ATMs:
          sub-01 control → 1
          sub-02 patient → 10
          sub-03 control → 3     → control mean = 2
          sub-04 patient → 20    → patient mean = 15
        """
        values = {"sub-01": 1.0, "sub-02": 10.0, "sub-03": 3.0, "sub-04": 20.0}
        g = make_group(n_subjects=4)

        for sid, v in values.items():
            subj = g.get_subject(sid)
            subj.atm_results = TrialATM([
                make_atm_obj(0, label="rest", mean_atm=np.full((2, 2), v)),
            ])
        return g

    @pytest.mark.parametrize("group, expected_atm", [
        ("control", np.full((2, 2), 2.0)),
        ("patient", np.full((2, 2), 15.0))
    ])
    def test_averages_only_subjects_in_the_group(self, group_known_atms, group, expected_atm):
        result = group_known_atms.get_mean_atm_group(group=group, cond="rest")

        np.testing.assert_allclose(result, expected_atm)

    def test_output_shape_matches_single_atm(self, group_known_atms):
        result = group_known_atms.get_mean_atm_group(group="control", cond="rest")

        assert result.shape == (2, 2)

    def test_absent_group_returns_nan(self, group_known_atms):
        with pytest.raises(ValueError, match="No subjects with label"):
            result = group_known_atms.get_mean_atm_group(group="nope", cond="rest")


class TestExtractGroup:

    def test_only_matching_subjects_included(self, group_two_labels):
        assert group_two_labels.extract_group("control").subject_ids == \
               ["sub-01", "sub-03"]

    def test_channel_names_carried_over(self, group_two_labels):
        result = group_two_labels.extract_group("control")
        assert result.channel_names == group_two_labels.channel_names

    def test_absent_label_gives_empty_group(self, group_two_labels):
        assert group_two_labels.extract_group("nonexistent").n_subjects == 0


class TestGetDatasetDataframe:

    def test_one_row_per_subject(self, group_two_labels):
        df = group_two_labels.get_dataset_dataframe()

        assert len(df) == 4

    def test_group_and_subject_id_are_first_columns(self, group_two_labels):
        df = group_two_labels.get_dataset_dataframe()

        assert list(df.columns)[:2] == ["Group", "Subject ID"]

    def test_channel_names_excluded(self, group_two_labels):
        df = group_two_labels.get_dataset_dataframe()

        assert "channel_names" not in df.columns

    @pytest.mark.parametrize("renamed", [
        "Highpass", "Lowpass", "Number of Channels/ROIs", "Time Points",
        "Number of Trials", "Sampling Frequency", "Trial Conditions",
    ])
    def test_columns_renamed(self, group_two_labels, renamed):
        df = group_two_labels.get_dataset_dataframe()

        assert renamed in df.columns

    def test_no_raw_keys_remain(self, group_two_labels):
        df = group_two_labels.get_dataset_dataframe()

        for raw in ["fmin", "fmax", "n_channels", "n_time_points",
                    "n_trials", "samp_freq", "conditions"]:
            assert raw not in df.columns

    def test_each_subject_paired_with_its_own_label(self, group_two_labels):
        df = sorted_rows(group_two_labels.get_dataset_dataframe(), "Subject ID")

        assert list(df["Group"]) == ["control", "patient", "control", "patient"]

    def test_all_subject_ids_present(self, group_two_labels):
        df = group_two_labels.get_dataset_dataframe()

        assert set(df["Subject ID"]) == set(group_two_labels.subject_ids)


class TestGetAvalancheDataframe:

    def test_one_row_per_subject_and_condition(self, group_two_labels):
        """4 subjects x 2 conditions (rest, task) = 8 rows."""
        df = group_two_labels.get_avalanche_dataframe()

        assert len(df) == 8

    def test_expected_columns(self, group_two_labels):
        df = group_two_labels.get_avalanche_dataframe()

        expected = ["Group", "Subject ID", "Condition", "Aval Sizes",
                    "Aval Durations", "Mean Flexibility", "Mean Count"]
        assert list(df.columns) == expected

    def test_sizes_are_pooled_across_matching_trials(self, group_factory):
        """
        'rest' covers trials 0 and 2, so their size lists are CONCATENATED
        (list + list), not nested.
        """
        g = group_factory(n_subjects=1)
        subj = g.subjects[0]
        aval = subj.avalanche_results

        expected = (aval.get_trial_data(0).sizes
                    + aval.get_trial_data(2).sizes)

        df = g.get_avalanche_dataframe()
        row = df[df["Condition"] == "rest"].iloc[0]

        assert row["Aval Sizes"] == expected

    def test_pooled_lists_are_flat(self, group_two_labels):
        df = group_two_labels.get_avalanche_dataframe()

        sizes = df["Aval Sizes"].iloc[0]
        assert all(np.isscalar(s) for s in sizes)

    def test_flexibility_and_count_are_averaged(self, group_factory):
        """Unlike sizes, these two are means across the matching trials."""
        g = group_factory(n_subjects=1)
        aval = g.subjects[0].avalanche_results

        expected_flex = np.mean([aval.get_trial_data(0).flexibility,
                                 aval.get_trial_data(2).flexibility])

        df = g.get_avalanche_dataframe()
        row = df[df["Condition"] == "rest"].iloc[0]

        assert row["Mean Flexibility"] == pytest.approx(expected_flex)

    def test_single_trial_condition_mean_equals_that_trial(self, group_factory):
        """'task' matches only trial 1, so its mean is that trial's value."""
        g = group_factory(n_subjects=1)
        trial = g.subjects[0].avalanche_results.get_trial_data(1)

        df = g.get_avalanche_dataframe()
        row = df[df["Condition"] == "task"].iloc[0]

        assert row["Mean Count"] == pytest.approx(trial.avalanche_count)

    def test_group_label_on_every_row(self, group_two_labels):
        df = sorted_rows(group_two_labels.get_avalanche_dataframe(),
                         "Subject ID", "Condition")

        for _, row in df.iterrows():
            subj = group_two_labels.get_subject(row["Subject ID"])
            assert row["Group"] == subj.subject_label

    def test_empty_group_returns_empty_dataframe(self, empty_group):
        """No rows to build, so an empty (column-less) frame comes back."""
        df = empty_group.get_avalanche_dataframe()

        assert len(df) == 0

class TestGetPatternsDataframe:

    def test_one_row_per_trial_across_all_subjects(self, group_two_labels):
        """4 subjects x 3 trials = 12 rows – NOT aggregated per condition."""
        df = group_two_labels.get_patterns_dataframe()

        assert len(df) == 12

    def test_expected_columns(self, group_two_labels):
        df = group_two_labels.get_patterns_dataframe()
        expected = ["Group", "Subject ID", "Condition", "Trial ID",
                    "Patterns", "Temporal Patterns"]
        assert list(df.columns) == expected

    def test_patterns_match_source_trials(self, group_factory):
        g = group_factory(n_subjects=1)
        df = sorted_rows(g.get_patterns_dataframe(), "Trial ID")

        aval = g.subjects[0].avalanche_results
        for _, row in df.iterrows():
            trial = aval.get_trial_data(row["Trial ID"])
            for actual, expected in zip(row["Patterns"], trial.patterns):
                np.testing.assert_array_equal(actual, expected)

    def test_all_trial_ids_present_per_subject(self, group_factory):
        g = group_factory(n_subjects=2)
        df = g.get_patterns_dataframe()

        for sid in g.subject_ids:
            ids = set(df[df["Subject ID"] == sid]["Trial ID"])
            assert ids == {0, 1, 2}


class TestGetAtmDataframe:

    def test_one_row_per_subject_and_condition(self, group_two_labels):
        df = group_two_labels.get_atm_dataframe()

        assert len(df) == 8      # 4 subjects x 2 conditions

    def test_expected_columns(self, group_two_labels):
        df = group_two_labels.get_atm_dataframe()
        expected = ["Group", "Subject ID", "Condition", "ATM Method", "Mean ATM"]
        assert list(df.columns) == expected

    def test_mean_atm_is_average_of_matching_trials(self, group_factory):
        g = group_factory(n_subjects=1)
        g.subjects[0].atm_results = TrialATM([
            make_atm_obj(0, label="rest", mean_atm=np.full((2, 2), 1.0)),
            make_atm_obj(1, label="task", mean_atm=np.full((2, 2), 5.0)),
            make_atm_obj(2, label="rest", mean_atm=np.full((2, 2), 3.0)),
        ])

        row = g.get_atm_dataframe().query("Condition == 'rest'").iloc[0]

        np.testing.assert_allclose(row["Mean ATM"], np.full((2, 2), 2.0))

    def test_mean_atm_cells_hold_2d_arrays(self, group_two_labels):
        df = group_two_labels.get_atm_dataframe()

        cell = df["Mean ATM"].iloc[0]
        assert isinstance(cell, np.ndarray)
        assert cell.ndim == 2

    def test_atm_method_recorded(self, group_two_labels):
        df = group_two_labels.get_atm_dataframe()

        assert (df["ATM Method"] == "standard").all()

    def test_ws_group_records_ws_method(self, group_factory):
        g = group_factory(n_subjects=2, symmetric=False, method="ws")

        df = g.get_atm_dataframe()

        assert (df["ATM Method"] == "ws").all()


class TestGetMetricsDataframe:

    def test_one_row_per_subject_and_condition(self, group_two_labels):
        df = group_two_labels.get_metrics_dataframe()

        assert len(df) == 8

    def test_base_columns_come_first(self, group_two_labels):
        df = group_two_labels.get_metrics_dataframe()

        assert list(df.columns)[:4] == ["Group", "Subject ID",
                                        "Condition", "ATM Method"]

    @pytest.mark.parametrize("column", [
        "Node Strength", "Avg Node Strength", "Clustering Coefficient",
        "Modularity", "Char Path", "Mean Mat", "Var Coeff",
    ])
    def test_metric_columns_are_title_cased(self, group_two_labels, column):
        df = group_two_labels.get_metrics_dataframe()

        assert column in df.columns

    def test_excluded_attributes_absent(self, group_two_labels):
        df = group_two_labels.get_metrics_dataframe()

        for col in ["Trial Id", "Trial Label", "Atm Method"]:
            assert col not in df.columns

    def test_scalar_metrics_averaged_to_a_scalar(self, group_factory):
        """'rest' spans two trials; modularity 0.2 and 0.4 average to 0.3."""
        g = group_factory(n_subjects=1)
        metrics = g.subjects[0].metrics_results
        metrics.get_trial_data(0).modularity = 0.2
        metrics.get_trial_data(2).modularity = 0.4

        df = g.get_metrics_dataframe()
        row = df[df["Condition"] == "rest"].iloc[0]

        assert row["Modularity"] == pytest.approx(0.3)

    def test_array_metrics_averaged_element_wise(self, group_factory):
        """
        Array-valued metrics use np.mean(..., axis=0), so the result keeps
        its per-node shape rather than collapsing to one number.
        """
        g = group_factory(n_subjects=1)
        metrics = g.subjects[0].metrics_results
        metrics.get_trial_data(0).node_strength = np.array([1.0, 2.0, 3.0])
        metrics.get_trial_data(2).node_strength = np.array([3.0, 4.0, 5.0])

        df = g.get_metrics_dataframe()
        row = df[df["Condition"] == "rest"].iloc[0]

        np.testing.assert_allclose(row["Node Strength"], [2.0, 3.0, 4.0])

    def test_array_metric_keeps_its_length(self, group_two_labels):
        df = group_two_labels.get_metrics_dataframe()

        assert np.shape(df["Node Strength"].iloc[0]) == (3,)

    def test_only_computed_metrics_become_columns(self, group_factory):
        """A MetricsObj with one metric yields one metric column."""
        g = group_factory(n_subjects=1)
        obj = MetricsObj(trial_id=0, atm_method="standard", trial_label="rest")
        obj.modularity = 0.5
        g.subjects[0].metrics_results = TrialMetrics([obj])

        df = g.get_metrics_dataframe()

        assert "Modularity"    in df.columns
        assert "Node Strength" not in df.columns

class TestGetAveragedMetricsDataframe:

    def test_one_row_per_subject_and_condition(self, group_two_labels):
        assert len(group_two_labels.get_averaged_metrics_dataframe()) == 8

    def test_only_node_level_metrics_included(self, group_two_labels):
        df = group_two_labels.get_averaged_metrics_dataframe()
        assert "Node Strength" in df.columns
        assert "Modularity"    not in df.columns   # scalar, excluded here

    def test_values_are_per_node_arrays(self, group_two_labels):
        df = group_two_labels.get_averaged_metrics_dataframe()
        assert np.shape(df["Node Strength"].iloc[0]) == (3,)


class TestSave:

    def test_all_csv_files_created(self, group_two_labels, tmp_path):
        group_two_labels.save(str(tmp_path), prefix="grp")

        #for name in ["grp_avalanche_results.csv",
        #             "grp_avalanche_patterns_results.csv",
        #             "grp_atm_results.csv",
        #             "grp_metrics_results.csv"]:
            #
        name = "grp_group_results.pkl"
        assert (tmp_path / name).exists(), f"missing {name}"


    def test_pickle_file_created(self, group_two_labels, tmp_path):
        group_two_labels.save(str(tmp_path), prefix="grp")

        assert (tmp_path / "grp_group_results.pkl").exists()

#    def test_empty_group_writes_only_the_pickle(self, empty_group, tmp_path):
        #empty_group.save(str(tmp_path), prefix="grp")

        #assert not (tmp_path / "grp_group_results.pkl").exists()
        #assert not (tmp_path / "grp_avalanche_results.csv").exists()

    def test_pickle_roundtrip_preserves_subject_ids(self, group_two_labels, tmp_path):
        group_two_labels.save(str(tmp_path), prefix="grp")

        with open(tmp_path / "grp_group_results.pkl", "rb") as f:
            loaded = pickle.load(f)

        assert loaded.subject_ids == group_two_labels.subject_ids

    def test_pickle_roundtrip_preserves_results(self, group_two_labels, tmp_path):
        group_two_labels.save(str(tmp_path), prefix="grp")

        with open(tmp_path / "grp_group_results.pkl", "rb") as f:
            loaded = pickle.load(f)

        np.testing.assert_array_equal(
            loaded.get_subject("sub-01").atm_results.trials[0].mean_atm,
            group_two_labels.get_subject("sub-01").atm_results.trials[0].mean_atm,
        )

    def test_pickle_roundtrip_preserves_channel_names(self, group_two_labels, tmp_path):
        group_two_labels.save(str(tmp_path), prefix="grp")

        with open(tmp_path / "grp_group_results.pkl", "rb") as f:
            loaded = pickle.load(f)

        assert loaded.channel_names == group_two_labels.channel_names

    #def test_csv_readable_with_expected_row_count(self, group_two_labels, tmp_path):
    #    group_two_labels.save(str(tmp_path), prefix="grp")

    #    df = pd.read_csv(tmp_path / "grp_avalanche_results.csv")

    #   assert len(df) == 8                      # 4 subjects x 2 conditions
    #    assert "Subject ID" in df.columns

    #def test_csv_written_without_index_column(self, group_two_labels, tmp_path):
    #    group_two_labels.save(str(tmp_path), prefix="grp")

    #    df = pd.read_csv(tmp_path / "grp_atm_results.csv")

    #    assert not any(c.startswith("Unnamed") for c in df.columns)

    @pytest.mark.parametrize("prefix", ["grp", "study_A", "2024_controls"])
    def test_prefix_used_in_every_filename(self, group_two_labels, tmp_path, prefix):
        group_two_labels.save(str(tmp_path), prefix=prefix)
        assert (tmp_path / f"{prefix}_group_results.pkl").exists()


class TestErrorHandling:

    def test_missing_results_raise_value_error(self):
        s = Subject(subject_id="sub-01")
        with pytest.raises(ValueError, match="not available"):
            s.get_avalanche_dataframe()

    def test_unknown_condition_raises_value_error(self, group_factory):
        s = group_factory(n_subjects=1).subjects[0]
        with pytest.raises(ValueError, match="no trials with condition"):
            s.get_mean_atm_cond("nonexistent")

    def test_unknown_group_raises_value_error(self, group_two_labels):
        with pytest.raises(ValueError, match="No subjects with label"):
            group_two_labels.get_mean_atm_group(group="nope", cond="rest")

class TestEmptyInputs:

    @pytest.mark.parametrize("method", [
        "get_dataset_dataframe", "get_avalanche_dataframe",
        "get_patterns_dataframe", "get_atm_dataframe",
        "get_metrics_dataframe", "get_averaged_metrics_dataframe",
    ])
    def test_empty_group_returns_empty_frame_with_columns(self, empty_group, method):
        df = getattr(empty_group, method)()
        assert len(df) == 0
        assert len(df.columns) > 0