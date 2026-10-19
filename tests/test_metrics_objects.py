import pickle
import numpy as np
import pytest

from pyAvalanches.connectivity_metrics import MetricsObj, TrialMetrics

def make_trial(trial_id, atm_method, trial_label, dummy_metric):
    """
    Builds a fully populated MetricsObj with known inputs and dummy metric
    """

    metrics_obj = MetricsObj(
    trial_id = trial_id,
    atm_method = atm_method,
    trial_label = trial_label
    )

    metrics_obj.dummy_metric = dummy_metric
    return metrics_obj


@pytest.fixture
def minimal_obj():
    """MetricsObj built with only the required arguments."""
    return MetricsObj(
        trial_id=0,
        atm_method='standard',
    )

@pytest.fixture
def full_obj():
    """MetricsObj with every optional argument explicitly set and dummy metric added."""

    metric_obj = MetricsObj(
        trial_id=0,
        atm_method='standard',
        trial_label="rest"
    )

    metric_obj.dummy_metric = np.zeros((1,5))
    return metric_obj

@pytest.fixture
def trial_metric():
    """
    TrialMetrics with 3 trials of known inputs.
    """
    return TrialMetrics([
        make_trial(trial_id=0, atm_method='standard', trial_label=0, dummy_metric=np.array([1,2,3])),
        make_trial(trial_id=1, atm_method='ws', trial_label=1, dummy_metric=np.array([4,5,6])),
        make_trial(trial_id=2, atm_method='standard', trial_label=0, dummy_metric=np.array([7,8,9])),
    ])

class TestMetricsObjInit:

    def test_required_arguments_stored_as_attributes(self, minimal_obj):
        assert minimal_obj.trial_id == 0
        assert minimal_obj.atm_method == 'standard'

    def test_optional_attributes_default_correctly(self, minimal_obj):
        assert minimal_obj.trial_label is None

    def test_all_arguments_stored_correctly(self, full_obj):
        assert full_obj.trial_id == 0
        assert full_obj.atm_method == 'standard'
        assert full_obj.trial_label == 'rest'
        np.testing.assert_array_equal(full_obj.dummy_metric, np.zeros((1,5)))

    @pytest.mark.parametrize("label", [0, 1, "task", "rest"])
    def test_trial_label_accepts_int_and_str(self, label):
        obj = MetricsObj(
            trial_id=0,
            atm_method='standard',
            trial_label=label
        )

        assert obj.trial_label == label

class TestTrialMetricsInit:

    def test_trials_stored_correctly(self, trial_metric):
        assert len(trial_metric.trials) == 3

    def test_no_argument_creates_empty_list(self):
        assert TrialMetrics().trials == []

class TestTrialMetricsProperties:

    def test_trial_labels_returns_correct_values(self, trial_metric):
        assert trial_metric.trial_labels == [0, 1, 0]

    def test_trial_indexes_returns_all_ids(self, trial_metric):
        assert trial_metric.trial_indexes == [0, 1, 2]

    def test_trial_count_returns_correct_value(self, trial_metric):
        assert trial_metric.trial_count == 3

class TestGetTrialData:

    @pytest.mark.parametrize("trial_idx, dummy_metric", [
        (0, [1,2,3]),
        (1, [4,5,6]),
        (2, [7,8,9])
    ])
    def test_returns_correct_trial(self, trial_metric, trial_idx, dummy_metric):
        assert trial_metric.get_trial_data(trial_idx).trial_id == trial_idx
        np.testing.assert_array_equal(trial_metric.get_trial_data(trial_idx).dummy_metric, dummy_metric)

    def test_missing_id_raises_value_error(self, trial_metric):
        with pytest.raises(ValueError, match="Trial ID"):
            trial_metric.get_trial_data(999)

class TestSave:

    def test_file_is_created(self, trial_metric, tmp_path):
        trial_metric.save(str(tmp_path), prefix="subj01")
        assert (tmp_path / "subj01_metrics_obj.pkl").exists()

    def test_saved_file_loads_without_error(self, trial_metric, tmp_path):
        """
        The most basic I/O check: the file must be valid pickle
        that can be opened and deserialized without raising.
        """
        trial_metric.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_metrics_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, TrialMetrics)

    def test_roundtrip_preserves_trial_count(self, trial_metric, tmp_path):
        """
        Round-trip test: save → load → compare.
        If serialization is lossy, the count would change.
        """
        original_count = trial_metric.trial_count
        trial_metric.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_metrics_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_count == original_count

    def test_roundtrip_preserves_trial_labels(self, trial_metric, tmp_path):
        original_labels = trial_metric.trial_labels
        trial_metric.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_metrics_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_labels == original_labels

    def test_correct_prefix_produces_correct_filename(
        self, trial_metric, tmp_path
    ):
        trial_metric.save(str(tmp_path), prefix="my_prefix")
        assert     (tmp_path / "my_prefix_metrics_obj.pkl").exists()
        assert not (tmp_path / "wrong_prefix_metrics_obj.pkl").exists()

    @pytest.mark.parametrize("prefix", ["subj01", "run_A", "2024_05_session1"])
    def test_various_prefixes_produce_correct_filename(
        self, trial_metric, tmp_path, prefix
    ):
        trial_metric.save(str(tmp_path), prefix=prefix)
        assert (tmp_path / f"{prefix}_metrics_obj.pkl").exists()