import pickle
import numpy as np
import pytest

from pyAvalanches.transition_matrices import ATMObj, TrialATM

def make_trial(trial_id, avalanche_transition_matrices, symmetric,
               method, mean_atm: np.ndarray, trial_label = None):
    """
        Builds a fully populated ATMObj with known inputs.
        """
    return ATMObj(
        trial_id=trial_id,
        avalanche_transition_matrices=avalanche_transition_matrices,
        symmetric=symmetric,
        method=method,
        mean_atm=mean_atm,
        trial_label=trial_label
    )

@pytest.fixture
def minimal_obj():
    """ATMObj built with only the required arguments."""
    return ATMObj(
        trial_id=0,
        avalanche_transition_matrices=[np.zeros((5,5)), np.zeros((5,5))],
        symmetric=True,
        method='standard',
        mean_atm=np.zeros((5,5)),
    )

@pytest.fixture
def full_obj():
    """ATMObj with every optional argument explicitly set."""
    return ATMObj(
        trial_id=0,
        avalanche_transition_matrices=[np.zeros((5,5)), np.zeros((5,5))],
        symmetric=True,
        method='standard',
        mean_atm=np.zeros((5,5)),
        trial_label="task"
    )

@pytest.fixture
def trial_atm():
    """
    TrialATM with 3 trials of known inputs.
    """
    return TrialATM([
        make_trial(trial_id=0,
                   avalanche_transition_matrices=[np.array([[1, 2],[3, 4]]), np.array([[1, 2],[3, 4]])],
                   symmetric=False,
                   method="standard",
                   mean_atm=np.array([[1, 2],[3, 4]]),
                   trial_label="rest"),
        make_trial(trial_id=1,
                   avalanche_transition_matrices=[np.array([[3, 1],[2, 3]]), np.array([[3, 1],[2, 3]])],
                   symmetric=False,
                   method="standard",
                   mean_atm=np.array([[3, 1],[2, 3]]),
                   trial_label="task"),
        make_trial(trial_id=2,
                   avalanche_transition_matrices=[np.array([[5, 6],[7, 8]]), np.array([[5, 6],[7, 8]])],
                   symmetric=False,
                   method="standard",
                   mean_atm=np.array([[5, 6],[7, 8]]),
                   trial_label="rest")
    ])

class TestATMObjInit:

    def test_required_arguments_stored_as_attributes(self, minimal_obj):
        assert minimal_obj.trial_id == 0
        np.testing.assert_array_equal(minimal_obj.avalanche_transition_matrices, [np.zeros((5, 5)), np.zeros((5, 5))])
        assert minimal_obj.symmetric == True
        assert minimal_obj.method == 'standard'
        np.testing.assert_array_equal(minimal_obj.mean_atm, np.zeros((5, 5)))

    def test_optional_attributes_default_correctly(self, minimal_obj):
        assert minimal_obj.trial_label is None

    def test_all_arguments_stored_correctly(self, full_obj):
        assert full_obj.trial_id == 0
        np.testing.assert_array_equal(full_obj.avalanche_transition_matrices, [np.zeros((5, 5)), np.zeros((5, 5))])
        assert full_obj.symmetric == True
        assert full_obj.method == 'standard'
        np.testing.assert_array_equal(full_obj.mean_atm, np.zeros((5, 5)))
        assert full_obj.trial_label == "task"

    @pytest.mark.parametrize("label", [0, 1, "task", "rest"])
    def test_trial_label_accepts_int_and_str(self, label):
        obj = ATMObj(
        trial_id=0,
        avalanche_transition_matrices=[np.zeros((5,5)), np.zeros((5,5))],
        symmetric=True,
        method='standard',
        mean_atm=np.zeros((5,5)),
        trial_label=label
        )
        assert obj.trial_label == label

class TestTrialATMInit:

    def test_trials_stored_correctly(self, trial_atm):
        assert len(trial_atm.trials) == 3

    def test_no_argument_creates_empty_list(self):
        assert TrialATM().trials == []

class TestTrialATMProperties:

    def test_trial_labels_returns_correct_values(self, trial_atm):
        assert trial_atm.trial_labels == ["rest", "task", "rest"]

    def test_trial_indexes_returns_all_ids(self, trial_atm):
        assert trial_atm.trial_indexes == [0, 1, 2]

    def test_trial_count_returns_correct_value(self, trial_atm):
        assert trial_atm.trial_count == 3

class TestAddTrial:

    def test_trial_is_added(self, trial_atm, full_obj):
        count_before = trial_atm.trial_count
        trial_atm.add_trial(full_obj)
        assert trial_atm.trial_count == count_before + 1

    def test_added_trial_is_retrievable(self, trial_atm):
        new_trial = make_trial(
            trial_id=10,
            avalanche_transition_matrices=[np.zeros((5, 5)), np.zeros((5, 5))],
            symmetric=True,
            method='standard',
            mean_atm=np.zeros((5, 5)),
            trial_label="task"
        )
        trial_atm.add_trial(new_trial)
        assert trial_atm.get_trial_data(10) is new_trial

    def test_add_to_empty_collection(self):
        trial_atm = TrialATM()
        new_trial = make_trial(
            trial_id=10,
            avalanche_transition_matrices=[np.zeros((5, 5)), np.zeros((5, 5))],
            symmetric=True,
            method='standard',
            mean_atm=np.zeros((5, 5)),
            trial_label="task"
        )
        trial_atm.add_trial(new_trial)
        assert trial_atm.trial_count == 1

class TestGetTrialData:

    @pytest.mark.parametrize("trial_idx", [0,1,2])
    def test_returns_correct_trial(self, trial_atm, trial_idx):
        assert trial_atm.get_trial_data(trial_idx).trial_id == trial_idx

    def test_missing_id_raises_value_error(self, trial_atm):
        with pytest.raises(ValueError, match="Trial ID"):
            trial_atm.get_trial_data(999)

class TestGlobalMeanATM:

    def test_gloabl_mean_is_correctly_returned(self, trial_atm):
        expected_mean = np.array(([3,3],[4,5]))
        np.testing.assert_array_equal(trial_atm.global_mean_atm(), expected_mean)
        #assert trial_atm.global_mean_atm() == np.array(([2,3],[4,5]))

    def test_empty_collection(self):
        with pytest.raises(ValueError, match="TrialATM object is empty"):
            TrialATM().global_mean_atm()

class TestGetGlobalMeanByCondition:

    @pytest.mark.parametrize("label, expected_mean", [
        ("rest", np.array(([3,4],[5,6]))),
        ("task", np.array([[3,1],[2,3]]))
    ])
    def test_mean_by_cond_is_correctly_returned(self, trial_atm, label, expected_mean):
        np.testing.assert_array_equal(trial_atm.get_atm_mean_by_condition(cond=label), expected_mean)
        #assert trial_atm.global_mean_atm() == np.array(([2,3],[4,5]))

    def test_error_is_raised_condition_not_existong(self, trial_atm):
        with pytest.raises(ValueError, match="not found"):
            trial_atm.get_atm_mean_by_condition(cond="wrong_label")

# tmp_path is a built-in pytest fixture: a fresh temporary directory
class TestSave:

    def test_file_is_created(self, trial_atm, tmp_path):
        trial_atm.save(str(tmp_path), prefix="subj01")
        assert (tmp_path / "subj01_atm_obj.pkl").exists()

    def test_saved_file_loads_without_error(self, trial_atm, tmp_path):
        """
        The most basic I/O check: the file must be valid pickle
        that can be opened and deserialized without raising.
        """
        trial_atm.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_atm_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert isinstance(loaded, TrialATM)

    def test_roundtrip_preserves_trial_count(self, trial_atm, tmp_path):
        """
        Round-trip test: save → load → compare.
        If serialization is lossy, the count would change.
        """
        original_count = trial_atm.trial_count
        trial_atm.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_atm_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_count == original_count

    def test_roundtrip_preserves_trial_labels(self, trial_atm, tmp_path):
        original_labels = trial_atm.trial_labels
        trial_atm.save(str(tmp_path), prefix="subj01")
        with open(tmp_path / "subj01_atm_obj.pkl", "rb") as f:
            loaded = pickle.load(f)
        assert loaded.trial_labels == original_labels

    def test_correct_prefix_produces_correct_filename(
        self, trial_atm, tmp_path
    ):
        trial_atm.save(str(tmp_path), prefix="my_prefix")
        assert     (tmp_path / "my_prefix_atm_obj.pkl").exists()
        assert not (tmp_path / "wrong_prefix_atm_obj.pkl").exists()

    @pytest.mark.parametrize("prefix", ["subj01", "run_A", "2024_05_session1"])
    def test_various_prefixes_produce_correct_filename(
        self, trial_atm, tmp_path, prefix
    ):
        trial_atm.save(str(tmp_path), prefix=prefix)
        assert (tmp_path / f"{prefix}_atm_obj.pkl").exists()