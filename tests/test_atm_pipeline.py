import numpy as np
import pytest
import pyAvalanches
from unittest.mock import MagicMock, patch

from pyAvalanches.avalanches import TrialAvalanches, AvalancheObj
from pyAvalanches.transition_matrices import ATMPipeline

MODULE = "pyAvalanches.transition_matrices"

@pytest.fixture
def standard_pipeline():
    return ATMPipeline(method="standard", symm=True)


@pytest.fixture
def standard_not_symm_pipeline():
    return ATMPipeline(method="standard", symm=False)

@pytest.fixture
def ws_pipeline():
    return ATMPipeline(method="ws", symm=False)

@pytest.fixture
def ws_pipeline_symm():
    return ATMPipeline(method="ws", symm=True)

@pytest.fixture
def simple_bin_data():
    """
    Simple binary data.

    1 0 0
    0 1 0
    1 0 1

    Expected standard symm ATM:
    [[0, 0.5, 0], [0.5, 0, 0.75], [0, 0.75, 0]]

    Expected standard not symm ATM:
    [[0, 1, 0], [0, 0, 1], [0, 0.5, 0]]

    Expected wsATM:
    [[0, 1, 0], [0, 0, 1], [0, 1, 0]]

    """

    data = np.zeros((3, 3), dtype=int)
    data[0, 0] = 1
    data[1, 1] = 1
    data[2, 0] = 1
    data[2, 2] = 1
    return data

def make_aval_trial(trial_id, matrices, durations=None, label=None):
    """
    Mock of a single AvalancheObj.
    avalanche_count is derived from the number of matrices, so passing an
    empty list gives a trial with no avalanches.
    """
    t = MagicMock(spec=AvalancheObj)
    t.trial_id        = trial_id
    t.avalanches_mat  = matrices
    t.durations       = durations if durations is not None else [1] * len(matrices)
    t.avalanche_count = len(matrices)
    t.trial_label     = label
    return t

def make_avalanches(trials):
    """
    Mock of a TrialAvalanches object.
    get_trial_data is given a real lookup so that process() retrieves trials
    by their trial_id (not by list position).
    """
    av = MagicMock(spec=TrialAvalanches)
    av.trial_indexes = [t.trial_id for t in trials]
    av.trial_labels  = [t.trial_label for t in trials]
    av.get_trial_data = lambda idx: next(t for t in trials if t.trial_id == idx)
    return av

def simple_matrix():
    """A small valid avalanche matrix (2 regions, 2 time steps)."""
    return np.array([[1, 0],
                     [0, 1]])

@pytest.fixture
def avalanches_3trials():
    """
    3 trials, ids 0/1/2, labels rest/MI/rest, 2 avalanches each.
    """
    return make_avalanches([
        make_aval_trial(0, [simple_matrix(), simple_matrix()], [2, 3], label=0),
        make_aval_trial(1, [simple_matrix(), simple_matrix()], [4, 5], label=1),
        make_aval_trial(2, [simple_matrix(), simple_matrix()], [1, 2], label=0),
    ])

def make_atm_trial(trial_id, mean_atm):
    """Mock of a single ATMObj – transform() only reads .mean_atm."""
    t = MagicMock()
    t.trial_id = trial_id
    t.mean_atm = mean_atm
    return t

def make_atm_results(trials):
    """
    Mock of a TrialATM object.

    trial_count and get_trial_data are the only two things transform() uses.
    get_trial_data gets a real lookup so it behaves like production (returns
    by trial_id, and raises StopIteration when the id is missing).
    """
    res = MagicMock()
    res.trial_count   = len(trials)
    res.get_trial_data = lambda idx: next(t for t in trials if t.trial_id == idx)
    return res

@pytest.fixture
def atm_results_3trials():
    """
    Mocked process() output: 3 trials, ids 0/1/2, each with a distinct 2x2 mean ATM.
    Distinct values make it easy to verify that rows are not mixed up.
    """
    return make_atm_results([
        make_atm_trial(0, np.array([[1.0, 2.0], [3.0, 4.0]])),
        make_atm_trial(1, np.array([[5.0, 6.0], [7.0, 8.0]])),
        make_atm_trial(2, np.array([[9.0, 10.0], [11.0, 12.0]])),
    ])


class TestInit:

    @pytest.mark.parametrize("method", ["standard", "ws"])
    def test_valid_methods_accepted(self, method):
        p = ATMPipeline(method=method, symm=True)
        assert p.method == method

    @pytest.mark.parametrize("bad_method", ["Standard", "WS", "mean", "", None])
    def test_invalid_method_raises_value_error(self, bad_method):
        with pytest.raises(ValueError, match="method must be one of"):
            ATMPipeline(method=bad_method, symm=True)

    def test_all_arguments_stored_as_attributes(self):
        p = ATMPipeline(method="ws", symm=True, trial_label=1)
        assert p.method == "ws"
        assert p.symm is True
        assert p.trial_label == 1

    def test_trial_label_defaults_to_none(self, standard_pipeline):
        assert standard_pipeline.trial_label is None

class TestComputeATMStandard:

    def test_output_shape_is_n_regions_by_n_regions(self, standard_pipeline, simple_bin_data):
        n_regions = simple_bin_data.shape[0]
        atm = standard_pipeline._compute_atm_standard(simple_bin_data)
        assert atm.shape == (n_regions, n_regions)

    def test_known_transition_gives_correct_values_symm(self, standard_pipeline, simple_bin_data):
        expected_symm = np.array([[0, 0.5, 0], [0.5, 0, 0.75], [0, 0.75, 0]])
        atm = standard_pipeline._compute_atm_standard(simple_bin_data)
        np.testing.assert_array_equal(atm, expected_symm)

    def test_known_transition_gives_correct_values_not_symm(self, standard_not_symm_pipeline, simple_bin_data):
        expected_not_symm = np.array([[0, 1, 0], [0, 0, 1], [0, 0.5, 0]])
        atm = standard_not_symm_pipeline._compute_atm_standard(simple_bin_data)
        np.testing.assert_array_equal(atm, expected_not_symm)

    def test_all_zeros_input_returns_zero_matrix(self, standard_pipeline):
        bin_data = np.zeros((3, 10), dtype=int)
        atm = standard_pipeline._compute_atm_standard(bin_data)
        np.testing.assert_array_equal(atm, np.zeros((3, 3)))

    def test_symm_true_returns_symmetric_matrix(self, standard_pipeline):
        # Asymmetric input: only region 0 → region 1 transition exists.
        bin_data = np.array([[1, 0],
                             [0, 1]])
        atm = standard_pipeline._compute_atm_standard(bin_data)
        np.testing.assert_array_equal(atm, atm.T)

    def test_symm_true_equals_average_of_atm_and_transpose(
            self, standard_pipeline, simple_bin_data
    ):
        # Compute without symmetrization first, then check the formula manually.
        raw_pipeline = ATMPipeline(method="standard", symm=False)
        atm_raw = raw_pipeline._compute_atm_standard(simple_bin_data)
        atm_symm = standard_pipeline._compute_atm_standard(simple_bin_data)
        expected = (atm_raw + atm_raw.T) / 2
        np.testing.assert_array_almost_equal(atm_symm, expected)

class TestComputeATMWs:

    def test_output_shape_is_n_regions_by_n_regions(self, ws_pipeline, simple_bin_data):
        n_regions = simple_bin_data.shape[0]
        atm = ws_pipeline._compute_atm_ws(simple_bin_data)
        assert atm.shape == (n_regions, n_regions)

    def test_known_transition_gives_correct_values(self, ws_pipeline, simple_bin_data):
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]])
        atm = ws_pipeline._compute_atm_ws(simple_bin_data)
        np.testing.assert_array_equal(atm, expected)

    def test_true_symm_input_is_ignored(self, ws_pipeline_symm, simple_bin_data):
        expected = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]) # this atm is not symmetrical
        atm = ws_pipeline_symm._compute_atm_ws(simple_bin_data)
        np.testing.assert_array_equal(atm, expected)

    def test_inactive_region_gets_uniform_row(self, ws_pipeline):
        """
        A region that is never active has no outgoing transitions,
        so the fallback (uniform distribution) should be applied.
        """
        # Region 1 is never active, so it has no outgoing transitions.
        bin_data = np.array([[1, 1],
                             [0, 0]])
        atm = ws_pipeline._compute_atm_ws(bin_data)
        n_regions = bin_data.shape[0]
        np.testing.assert_array_almost_equal(
            atm[1, :], np.full(n_regions, 1 / n_regions)
        )

class TestComputeStochasticMeanATM:

    @staticmethod
    def _make_inputs_atm(atm_list, durations):
        """
        Builds the two mock objects required by _compute_stochastic_mean_atm.
        """
        #ATMObj
        trial_atm_data = MagicMock()
        trial_atm_data.avalanche_transition_matrices = atm_list

        #AvalancheObj
        avalanche_data = MagicMock()
        avalanche_data.durations = durations

        return trial_atm_data, avalanche_data

    def test_output_shape_matches_input_atm_shape(self, ws_pipeline):
        atm_list = [np.eye(3), np.eye(3)]
        durations = [2, 3]
        trial_atm_data, avalanche_data = self._make_inputs_atm(atm_list, durations)

        ws_atm = ws_pipeline._compute_stochastic_mean_atm(trial_atm_data, avalanche_data)

        assert ws_atm.shape == (3,3)

    def test_equal_durations_give_simple_mean(self, ws_pipeline):
        """
        When all durations are equal the weighted mean collapses to a
        simple arithmetic mean.
        """
        atm1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        atm2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        trial_atm_data, avalanche_data = self._make_inputs_atm(
            [atm1, atm2], durations=[2, 2]
        )

        result = ws_pipeline._compute_stochastic_mean_atm(trial_atm_data, avalanche_data)
        expected = (atm1 + atm2) / 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_known_values_produce_correct_result(self, ws_pipeline):
        """Manual calculation with small matrices."""
        atm1 = np.array([[1.0, 0.0], [0.0, 1.0]])  # duration 2
        atm2 = np.array([[0.0, 1.0], [1.0, 0.0]])  # duration 3
        trial_atm_data, avalanche_data = self._make_inputs_atm(
            [atm1, atm2], durations=[2, 3]
        )

        result = ws_pipeline._compute_stochastic_mean_atm(trial_atm_data, avalanche_data)

        # (2*atm1 + 3*atm2) / 5
        expected = (2 * atm1 + 3 * atm2) / 5
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_avalanche_returns_that_atm(self, ws_pipeline):
        """With only one avalanche the weighted mean is just the ATM itself."""
        atm = np.array([[0.5, 0.5], [0.3, 0.7]])
        trial_atm_data, avalanche_data = self._make_inputs_atm([atm], durations=[4])

        result = ws_pipeline._compute_stochastic_mean_atm(trial_atm_data, avalanche_data)

        np.testing.assert_array_almost_equal(result, atm)


class TestProcess:

    @pytest.mark.parametrize("wrong_input", [
        {"key": "value"},  # dict
        [1, 2, 3],  # list
        42,  # int
        "signal",  # str
        3.14,  # float
    ])
    def test_error_unsupported_type(self, standard_pipeline, wrong_input):
        with pytest.raises(TypeError, match="must be a TrialAvalanches object"):
            standard_pipeline.process(wrong_input)


    def test_absent_label_raises_value_error(self, avalanches_3trials):
        p = ATMPipeline(method='standard', symm = True, trial_label="wrong_label")
        with pytest.raises(ValueError, match="No trials found with label"):
            p.process(avalanches_3trials)

    @pytest.mark.parametrize("label, label_list, count, index", [
        (None, [0, 1, 0], 3, [0, 1, 2]),  # no label
        (1, [1], 1, [1]),  # valid label for one trial
        (0, [0, 0], 2, [0, 2])  # valid label for two trials
    ])
    def test_label_selection(self, avalanches_3trials,
                             label, label_list, count, index):

        p = ATMPipeline(method='standard', symm = True, trial_label=label)
        results = p.process(avalanches_3trials)
        assert results.trial_labels == label_list
        assert results.trial_count == count
        assert results.trial_indexes == index

    def test_standard_method_routes_to_compute_atm_standard(
            self, standard_pipeline, avalanches_3trials
    ):
        """
        3 trials x 2 avalanches = 6 expected calls of _compute_atm_standard.
        """
        standard_pipeline._compute_atm_standard = MagicMock(return_value=np.eye(2))
        standard_pipeline._compute_atm_ws = MagicMock(return_value=np.eye(2))

        standard_pipeline.process(avalanches_3trials)

        assert standard_pipeline._compute_atm_standard.call_count == 6
        assert standard_pipeline._compute_atm_ws.call_count == 0

    def test_ws_method_routes_to_compute_atm_ws(
            self, ws_pipeline, avalanches_3trials
    ):
        """
        3 trials x 2 avalanches = 6 expected calls of _compute_atm_ws.
        """
        ws_pipeline._compute_atm_standard = MagicMock(return_value=np.eye(2))
        ws_pipeline._compute_atm_ws = MagicMock(return_value=np.eye(2))
        ws_pipeline._compute_stochastic_mean_atm = MagicMock(return_value=np.eye(2))

        ws_pipeline.process(avalanches_3trials)

        assert ws_pipeline._compute_atm_ws.call_count == 6
        assert ws_pipeline._compute_atm_standard.call_count == 0

    def test_ws_method_calls_stochastic_mean_once_per_trial(
            self, ws_pipeline, avalanches_3trials
    ):
        """The weighted mean is computed per trial, not per avalanche."""
        ws_pipeline._compute_stochastic_mean_atm = MagicMock(return_value=np.eye(2))
        ws_pipeline.process(avalanches_3trials)

        assert ws_pipeline._compute_stochastic_mean_atm.call_count == 3

    def test_standard_method_never_calls_stochastic_mean(
            self, standard_pipeline, avalanches_3trials
    ):
        standard_pipeline._compute_stochastic_mean_atm = MagicMock(return_value=np.eye(2))
        standard_pipeline.process(avalanches_3trials)

        assert standard_pipeline._compute_stochastic_mean_atm.call_count == 0

    @pytest.mark.parametrize("method, symm", [
        ('standard', True),
        ('standard', False),
        ('ws', False)
    ])
    def test_all_pipelines_process_trial_correctly(self, avalanches_3trials, method, symm):
        p = ATMPipeline(method=method, symm=symm)
        results = p.process(avalanches_3trials)
        assert results.trial_count == 3            # number of trials
        assert results.trial_labels == [0,1,0]     # labels
        assert results.trial_indexes == [0,1,2]    # indexes
        assert len(results.trials) == 3            # number of ATMObj created
        for i in results.trial_indexes:
            assert len(results.trials[i].avalanche_transition_matrices) == 2
            assert results.trials[i].method == p.method
            assert results.trials[i].symmetric == p.symm

    def test_ws_method_forces_symmetric_false(self, avalanches_3trials):
        p = ATMPipeline(method='ws', symm=True)
        results = p.process(avalanches_3trials)
        for i in results.trial_indexes:
            assert results.trials[i].symmetric == False

    @patch(f"{MODULE}.TrialATM")
    @patch(f"{MODULE}.ATMObj")
    def test_ws_method_overwrites_mean_atm_with_weighted_mean(
            self, mock_atm, mock_ta, avalanches_3trials
    ):
        """
        For the ws method, the mean_atm passed to the constructor is replaced
        afterwards by the output of _compute_stochastic_mean_atm.
        """
        result_mock = MagicMock()
        mock_atm.return_value = result_mock

        weighted_mean = np.full((2, 2), 7.0)
        p = ATMPipeline(method="ws", symm=False)
        p._compute_stochastic_mean_atm = MagicMock(return_value=weighted_mean)

        p.process(avalanches_3trials)

        np.testing.assert_array_equal(result_mock.mean_atm, weighted_mean)

    def test_trial_without_avalanches_prints_warning(
            self, standard_pipeline, capsys
    ):
        """capsys captures stdout so we can check the user-facing message."""
        avalanches = make_avalanches([make_aval_trial(0, [])])

        standard_pipeline.process(avalanches)

        assert "WARNING" in capsys.readouterr().out

    def test_empty_trial_skipped_but_others_still_processed(
            self, standard_pipeline
    ):
        """Trial 1 is empty; trials 0 and 2 must still produce results."""
        avalanches = make_avalanches([
            make_aval_trial(0, [simple_matrix()]),
            make_aval_trial(1, []),  # empty
            make_aval_trial(2, [simple_matrix()]),
        ])

        results = standard_pipeline.process(avalanches)

        assert results.trial_count == 2
        assert results.trial_indexes == [0,2]

    def test_all_trials_empty_gives_empty_trial_atm(self, standard_pipeline):
        """TrialATM must still be constructed, but with an empty list."""
        avalanches = make_avalanches([
            make_aval_trial(0, []),
            make_aval_trial(1, []),
        ])

        results = standard_pipeline.process(avalanches)
        assert results.trial_count == 0
        assert results.trial_indexes == []



class TestFit:

    def test_returns_self(self, standard_pipeline, avalanches_3trials):
        """
        sklearn.Pipeline chains calls on the return value of fit(), so
        returning anything other than self would silently break it.
        This is the ONE contract that matters for a stub fit().
        """
        assert standard_pipeline.fit(standard_pipeline) is standard_pipeline

    def test_does_not_mutate_any_attribute(self, standard_pipeline, avalanches_3trials):
        """
        A stub must have no side-effects.  We snapshot __dict__ before
        and compare after to detect any accidental state change.
        """
        snapshot = standard_pipeline.__dict__.copy()
        standard_pipeline.fit(avalanches_3trials)
        assert standard_pipeline.__dict__ == snapshot

    def test_accepts_optional_y_argument(self, standard_pipeline, avalanches_3trials):
        """
        sklearn convention: fit(X, y=None) must tolerate a y argument
        without raising, even if it ignores it.
        """
        assert standard_pipeline.fit(avalanches_3trials, y=np.zeros(3)) is standard_pipeline

class TestTransform:

    def test_returns_numpy_array(self, standard_pipeline, avalanches_3trials):
        results = standard_pipeline.transform(avalanches_3trials)
        assert isinstance(results, np.ndarray)

    def test_output_shape_is_correct_standard(self, standard_pipeline, avalanches_3trials):
        n_trials = 3
        n_regions = 2
        expected_dim = (n_trials, n_regions*n_regions)
        results = standard_pipeline.transform(avalanches_3trials)
        assert results.shape == expected_dim

    def test_output_shape_is_correct_ws(self, ws_pipeline, avalanches_3trials):
        n_trials = 3
        n_regions = 2
        expected_dim = (n_trials, n_regions*n_regions)
        results = ws_pipeline.transform(avalanches_3trials)
        assert results.shape == expected_dim

    def test_first_row_is_flattened_first_mean_atm(
            self, standard_pipeline, atm_results_3trials
    ):
        """
        np.reshape uses C (row-major) order, so [[1,2],[3,4]] → [1,2,3,4].
        """
        standard_pipeline.process = MagicMock(return_value=atm_results_3trials)

        result = standard_pipeline.transform("dummy_input")

        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 4.0])

    def test_rows_are_in_trial_id_order(self, standard_pipeline, atm_results_3trials):
        """Row 0 must come from trial 0, row 1 from trial 1, and so on."""
        standard_pipeline.process = MagicMock(return_value=atm_results_3trials)

        result = standard_pipeline.transform("dummy_input")

        assert result[0][0] == 1.0  # trial 0, element [0,0]
        assert result[1][0] == 5.0  # trial 1, element [0,0]
        assert result[2][0] == 9.0  # trial 2, element [0,0]

    def test_process_called_once(self, standard_pipeline, atm_results_3trials):
        standard_pipeline.process = MagicMock(return_value=atm_results_3trials)
        standard_pipeline.transform("dummy_input")
        standard_pipeline.process.assert_called_once()

    def test_row_equals_flattened_mean_atm_from_process(self, standard_pipeline):
        """
        Cross-check the two entry points: transform() row i must equal
        process()'s mean_atm for trial i, flattened.
        """
        avalanches = TrialAvalanches([
            make_aval_trial(0, [simple_matrix(), simple_matrix()]),
            make_aval_trial(1, [simple_matrix(), simple_matrix()]),
        ])

        result = standard_pipeline.transform(avalanches)
        process_output = standard_pipeline.process(avalanches)

        for i in range(2):
            expected = process_output.get_trial_data(i).mean_atm.flatten()
            np.testing.assert_array_almost_equal(result[i], expected)