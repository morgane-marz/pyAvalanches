import pickle
import numpy as np
import pytest

from pyAvalanches.avalanches import AvalancheObj, TrialAvalanches

def make_trial(trial_id, durations, sizes, label=None):
    """
    Builds a fully populated AvalancheObj with known durations/sizes/label.
    """
    n = len(durations)
    return AvalancheObj(
        trial_id=trial_id,
        avalanches_mat=[np.zeros((2, d)) for d in durations],
        time_bins=1,
        sizes=sizes,
        durations=durations,
        patterns=[np.zeros(2)] * n,
        patterns_temporal=[np.zeros(2)] * n,
        bounds=[(0, d - 1) for d in durations],
        flexibility=1,
        avalanche_count=n,
        trial_label=label,
    )

@pytest.fixture
def minimal_obj():
    """AvalancheObj built with only the required arguments."""
    return AvalancheObj(
        trial_id=0,
        avalanches_mat=[np.zeros((3, 5))]
    )

@pytest.fixture
def full_obj():
    """AvalancheObj with every optional argument explicitly set."""
    return AvalancheObj(
        trial_id=2,
        avalanches_mat=[np.zeros((3, 5)), np.ones((3, 3))],
        time_bins=2,
        sizes=[5, 9],
        durations=[3, 4],
        patterns=[np.array([1, 0, 1]), np.array([0, 1, 1])],
        patterns_temporal=[np.array([1, 0, 2]), np.array([0, 1, 3])],
        bounds=[(0, 2), (5, 8)],
        flexibility=2,
        avalanche_count=2,
        trial_label="task",
    )

@pytest.fixture
def trial_avalanches():
    """
    TrialAvalanches with 3 trials of known labels, durations and sizes.

    trial 0 : label='rest', durations=[2, 3], sizes=[4, 6]
    trial 1 : label='MI',   durations=[4, 5], sizes=[8, 10]
    trial 2 : label='rest', durations=[1, 2], sizes=[2, 3]

    """
    return TrialAvalanches([
        make_trial(0, durations=[2, 3], sizes=[4, 6],   label="rest"),
        make_trial(1, durations=[4, 5], sizes=[8, 10],  label="task"),
        make_trial(2, durations=[1, 2], sizes=[2, 3],   label="rest"),
    ])

class TestAvalancheObjInit:

    def test_required_arguments_stored_as_attributes(self, minimal_obj):
        assert minimal_obj.trial_id == 0
        #assert minimal_obj.time_bins == 1
        assert len(minimal_obj.avalanches_mat) == 1

    def test_optional_attributes_default_correctly(self, minimal_obj):
        for attr in ["sizes", "durations", "patterns", "patterns_temporal",
                     "bounds", "flexibility", "avalanche_count", "trial_label"]:
            assert getattr(minimal_obj, attr) is None
            assert minimal_obj.time_bins == 1

    def test_all_arguments_stored_correctly(self, full_obj):
        assert full_obj.trial_id == 2
        assert full_obj.time_bins == 2
        assert full_obj.sizes == [5, 9]
        assert full_obj.durations == [3, 4]
        assert full_obj.flexibility == 2
        assert full_obj.avalanche_count == 2
        assert full_obj.trial_label == "task"

    @pytest.mark.parametrize("label", [0, 1, "task", "rest"])
    def test_trial_label_accepts_int_and_str(self, label):
        obj = AvalancheObj(trial_id=0, avalanches_mat=[], trial_label=label)
        assert obj.trial_label == label

class TestTrialAvalanchesInit:

    def test_trials_stored_correctly(self, trial_avalanches):
        assert len(trial_avalanches.trials) == 3

    def test_no_argument_creates_empty_list(self):
        assert TrialAvalanches().trials == []

class TestTrialAvalanchesProperties:

    def test_trial_labels_returns_correct_values(self, trial_avalanches):
        assert trial_avalanches.trial_labels == ["rest", "task", "rest"]

    def test_trial_indexes_returns_all_ids(self, trial_avalanches):
        assert trial_avalanches.trial_indexes == [0, 1, 2]

    def test_trial_count_returns_correct_value(self, trial_avalanches):
        assert trial_avalanches.trial_count == 3

class TestAddTrial:

    def test_trial_is_added(self, trial_avalanches):
        count_before = trial_avalanches.trial_count
        trial_avalanches.add_trial(make_trial(10, [2], [4]))
        assert trial_avalanches.trial_count == count_before + 1

    def test_added_trial_is_retrievable(self, trial_avalanches):
        new_trial = make_trial(10, [2], [4])
        trial_avalanches.add_trial(new_trial)
        assert trial_avalanches.get_trial_data(10) is new_trial

    def test_add_to_empty_collection(self):
        trial_aval = TrialAvalanches()
        trial_aval.add_trial(make_trial(1, [2], [4]))
        assert trial_aval.trial_count == 1

class TestGetTrialData:

    @pytest.mark.parametrize("trial_idx", [0,1,2])
    def test_returns_correct_trial(self, trial_avalanches, trial_idx):
        assert trial_avalanches.get_trial_data(trial_idx).trial_id == trial_idx

    def test_missing_id_raises_value_error(self, trial_avalanches):
        with pytest.raises(ValueError, match="Trial ID"):
            trial_avalanches.get_trial_data(999)

class TestGetAllDurations:

    def test_merged_returns_flat_list(self, trial_avalanches):
        assert trial_avalanches.get_all_durations(merged=True) == [2, 3, 4, 5, 1, 2]

    def test_not_merged_returns_nested_list(self, trial_avalanches):
        assert trial_avalanches.get_all_durations(merged=False) == [[2, 3], [4, 5], [1, 2]]

    def test_empty_collection_merged(self):
        assert TrialAvalanches().get_all_durations(merged=True) == []

    def test_empty_collection_not_merged(self):
        assert TrialAvalanches().get_all_durations(merged=False) == []

class TestGetDurationsByCond:

    def test_merged_returns_flat_list(self, trial_avalanches):
        assert trial_avalanches.get_durations_by_cond("rest", merged=True) == [2, 3, 1, 2]

    def test_not_merged_returns_nested_list(self, trial_avalanches):
        assert trial_avalanches.get_durations_by_cond("rest", merged=False) == [[2, 3], [1, 2]]

    def test_single_matching_trial(self, trial_avalanches):
        assert trial_avalanches.get_durations_by_cond("task", merged=False) == [[4, 5]]

    def test_absent_condition_returns_empty_list(self, trial_avalanches):
        assert trial_avalanches.get_durations_by_cond("nonexistent", merged=True) == []
        assert trial_avalanches.get_durations_by_cond("nonexistent", merged=False) == []

class TestGetAllSizes:

    def test_merged_returns_flat_list(self, trial_avalanches):
        assert trial_avalanches.get_all_sizes(merged=True) == [4, 6, 8, 10, 2, 3]

    def test_not_merged_returns_nested_list(self, trial_avalanches):
        assert trial_avalanches.get_all_sizes(merged=False) == [[4, 6], [8, 10], [2, 3]]

    def test_empty_collection_merged(self):
        assert TrialAvalanches().get_all_sizes(merged=True) == []

    def test_empty_collection_not_merged(self):
        assert TrialAvalanches().get_all_sizes(merged=False) == []

class TestGetSizesByCond:

    def test_merged_returns_flat_list(self, trial_avalanches):
        assert trial_avalanches.get_sizes_by_cond("rest", merged=True) == [4, 6, 2, 3]

    def test_not_merged_returns_nested_list(self, trial_avalanches):
        assert trial_avalanches.get_sizes_by_cond("rest", merged=False) == [[4, 6], [2, 3]]

    def test_single_matching_trial(self, trial_avalanches):
        assert trial_avalanches.get_sizes_by_cond("task", merged=True) == [8, 10]

    def test_absent_condition_returns_empty_list(self, trial_avalanches):
        assert trial_avalanches.get_sizes_by_cond("nonexistent", merged=True) == []

class TestSave:

    def test_file_is_created(self, trial_avalanches, tmp_path):
        trial_avalanches.save(str(tmp_path), prefix="subj01")
        assert (tmp_path / "subj01_avalanche_obj.pkl").exists()

    def test_saved_file_loads_without_error(self, trial_avalanches, tmp_path):
        """
        The most basic I/O check: the file must be valid pickle
        that can be opened and deserialized without raising.
        """
        trial_avalanches.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_avalanche_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, TrialAvalanches)

    def test_roundtrip_preserves_trial_count(self, trial_avalanches, tmp_path):
        """
        Round-trip test: save → load → compare.
        If serialization is lossy, the count would change.
        """
        original_count = trial_avalanches.trial_count
        trial_avalanches.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_avalanche_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_count == original_count

    def test_roundtrip_preserves_trial_labels(self, trial_avalanches, tmp_path):
        original_labels = trial_avalanches.trial_labels
        trial_avalanches.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_avalanche_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_labels == original_labels

    def test_correct_prefix_produces_correct_filename(
        self, trial_avalanches, tmp_path
    ):
        trial_avalanches.save(str(tmp_path), prefix="my_prefix")
        assert     (tmp_path / "my_prefix_avalanche_obj.pkl").exists()
        assert not (tmp_path / "wrong_prefix_avalanche_obj.pkl").exists()

    @pytest.mark.parametrize("prefix", ["subj01", "run_A", "2024_05_session1"])
    def test_various_prefixes_produce_correct_filename(
        self, trial_avalanches, tmp_path, prefix
    ):
        trial_avalanches.save(str(tmp_path), prefix=prefix)
        assert (tmp_path / f"{prefix}_avalanche_obj.pkl").exists()