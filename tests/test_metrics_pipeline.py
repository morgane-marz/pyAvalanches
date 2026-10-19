import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pyAvalanches.connectivity_metrics import NetworkMetricsPipeline, TrialMetrics
from pyAvalanches.transition_matrices import  ATMObj, TrialATM

MODULE = "pyAvalanches.connectivity_metrics"

def symmetric_matrix():
    """
    3-node weighted symmetric matrix with zero diagonal.
    Valid input for bct and networkx, and fully connected so path-based
    metrics do not hit infinities.

    Column sums (= node strengths): [0.7, 1.3, 1.0]
    """
    return np.array([[0.0, 0.5, 0.2],
                     [0.5, 0.0, 0.8],
                     [0.2, 0.8, 0.0]])


def make_atm_trial(trial_id, mean_atm=None, symmetric=True,
                   method="standard", label=None):
    """
    Mock of a single ATMObj.
    process() reads only .symmetric, .method, .mean_atm and .trial_label.
    """
    t = MagicMock()
    t.trial_id    = trial_id
    t.mean_atm    = symmetric_matrix() if mean_atm is None else mean_atm
    t.symmetric   = symmetric
    t.method      = method
    t.trial_label = label
    return t


def make_atm_data(trials):
    """
    Mock of a TrialATM object.
    get_trial_data gets a real lookup so process() retrieves trials by ID,
    exactly as in production.
    """
    res = MagicMock()
    res.trial_indexes  = [t.trial_id for t in trials]
    res.trial_labels   = [t.trial_label for t in trials]
    res.trial_count    = len(trials)
    res.get_trial_data = lambda idx: next(t for t in trials if t.trial_id == idx)
    return res

def processed_ids(results):
    """Trial IDs that made it into the results, in order."""
    return [t.trial_id for t in results.trials]


def processed_labels(results):
    return [t.trial_label for t in results.trials]

@pytest.fixture
def symm_pipeline():
    """Symmetric pipeline asking for a single cheap metric."""
    return NetworkMetricsPipeline(metrics=["node_strength"], symm=True)


@pytest.fixture
def nonsymm_pipeline():
    return NetworkMetricsPipeline(metrics=["connectivity_measures"], symm=False)


@pytest.fixture
def atm_data_3symm():
    """3 trials, all symmetric, labels rest/MI/rest."""
    return make_atm_data([
        make_atm_trial(0, symmetric=True, label="rest"),
        make_atm_trial(1, symmetric=True, label="MI"),
        make_atm_trial(2, symmetric=True, label="rest"),
    ])

METRIC_SPECS = {
    "connectivity_measures": {
        "method":  "_compute_connectivity_measures",
        "returns": (0.11, 0.22, 0.33, np.array([1.0, 2.0, 3.0])),
        "attrs": {
            "mean_mat":  0.11,
            "var_mat":   0.22,
            "var_coeff": 0.33,
            "skew_mat":  np.array([1.0, 2.0, 3.0]),
        },
    },
    "clustering_coeff": {
        "method":  "_compute_clust_coeff",
        "returns": (np.array([0.4, 0.5, 0.6]), 0.5),
        "attrs": {
            "clustering_coefficient":     np.array([0.4, 0.5, 0.6]),
            "avg_clustering_coefficient": 0.5,
        },
    },
    "node_strength": {
        "method":  "_compute_node_strength",
        "returns": (np.array([0.7, 0.8, 0.9]), 0.8),
        "attrs": {
            "node_strength":     np.array([0.7, 0.8, 0.9]),
            "avg_node_strength": 0.8,
        },
    },
    "eigenvect_centrality": {
        "method":  "_compute_eigen_centr",
        "returns": (np.array([0.1, 0.6, 0.7]), 0.4666667),
        "attrs": {
            "eigenvector_centrality":     np.array([0.1, 0.6, 0.7]),
            "avg_eigenvector_centrality": 0.4666667,
        },
    },
    "modularity": {
        "method":  "_compute_modularity",
        "returns": 0.42,
        "attrs":   {"modularity": 0.42},
    },
    "char_path": {
        "method":  "_compute_charpath_len",
        "returns": 3.14,
        "attrs":   {"char_path": 3.14},
    },
}

ALL_METHOD_NAMES = [spec["method"] for spec in METRIC_SPECS.values()]

def stub_all_metrics(pipeline):
    """
    Replaces every metric computation on this ONE instance with a mock that
    returns the value declared in METRIC_SPECS.

    Assigning to the instance shadows the class-level staticmethod, so the
    call inside process() becomes plain mock(matrix) – meaning call_args[0][0]
    is the matrix, with no `self` to skip over.
    """
    for spec in METRIC_SPECS.values():
        setattr(pipeline, spec["method"], MagicMock(return_value=spec["returns"]))

class TestInit:

    @pytest.mark.parametrize("metric", [
        "clustering_coeff", "node_strength", "eigenvect_centrality",
        "modularity", "char_path", "connectivity_measures",
    ])
    def test_all_symmetric_metrics_accepted(self, metric):
        p = NetworkMetricsPipeline(metrics=[metric], symm=True)
        assert p.metrics == [metric]

    def test_multiple_symmetric_metrics_accepted(self):
        metrics = ["node_strength", "modularity", "char_path"]
        p = NetworkMetricsPipeline(metrics=metrics, symm=True)
        assert p.metrics == metrics

    def test_connectivity_measures_accepted_when_not_symmetric(self):
        p = NetworkMetricsPipeline(metrics=["connectivity_measures"], symm=False)
        assert p.metrics == ["connectivity_measures"]

    @pytest.mark.parametrize("bad_metric", [
        "clustering", "Node_strength", "degree", "", None,
    ])
    def test_invalid_metric_raises_when_symmetric(self, bad_metric):
        with pytest.raises(ValueError, match="undirected \\(symmetric\\) ATMs"):
            NetworkMetricsPipeline(metrics=[bad_metric], symm=True)

    @pytest.mark.parametrize("bad_metric", [
        "node_strength", "modularity", "char_path",
        "clustering_coeff", "eigenvect_centrality",
    ])
    def test_symmetric_only_metrics_rejected_when_not_symmetric(self, bad_metric):
        """
        These are all valid for symm=True, but only 'connectivity_measures'
        is allowed for directed ATMs.
        """
        with pytest.raises(ValueError, match="directed \\(non-symmetric\\) ATMs"):
            NetworkMetricsPipeline(metrics=[bad_metric], symm=False)

    def test_one_bad_metric_in_a_valid_list_still_raises(self):
        """Validation must check every entry, not just the first."""
        with pytest.raises(ValueError):
            NetworkMetricsPipeline(
                metrics=["node_strength", "modularity", "bad_metric"], symm=True
            )

    def test_symm_stored_as_attribute(self):
        assert NetworkMetricsPipeline(metrics=["node_strength"], symm=True).symm is True
        assert NetworkMetricsPipeline(metrics=["connectivity_measures"], symm=False).symm is False

    def test_empty_metrics_raises_value_error(self):
        with pytest.raises(ValueError, match="Empty metrics list was given"):
            NetworkMetricsPipeline(metrics=[], symm=True)


# TODO should I test the metrics methods? I'm calling external toolboxes so doesn't make sense imo

class TestProcess:

    def test_empty_input_raises_value_error(self, symm_pipeline):
        """An empty TrialATM is an error, not a silent empty result."""
        with pytest.raises(ValueError, match="contains no trials"):
            symm_pipeline.process(TrialATM([]))

    def test_empty_input_raises_for_non_symmetric_pipeline(self, nonsymm_pipeline):
        with pytest.raises(ValueError, match="contains no trials"):
            nonsymm_pipeline.process(TrialATM([]))

    @pytest.mark.parametrize("bad_value", [None, 1, 0, "True", "", [], 1.0])
    def test_non_boolean_symmetric_raises_type_error(self, symm_pipeline, bad_value):
        """
        Anything that is not a bool is rejected, including truthy/falsy values
        like 1 and 0.
        """
        atm_data = TrialATM([make_atm_trial(0, symmetric=bad_value)])

        with pytest.raises(TypeError, match="'symmetric' must be a boolean"):
            symm_pipeline.process(atm_data)

    def test_error_message_names_the_offending_trial(self, symm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(7, symmetric=None),
        ])

        with pytest.raises(TypeError, match="trial 7"):
            symm_pipeline.process(atm_data)

    def test_all_symmetric_trials_processed(self, symm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=True),
            make_atm_trial(2, symmetric=True),
        ])

        results = symm_pipeline.process(atm_data)

        assert processed_ids(results) == [0, 1, 2]

    def test_non_symmetric_trials_are_skipped(self, symm_pipeline):
        """Trial 1 is directed and must be dropped by a symm=True pipeline."""
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=False),
            make_atm_trial(2, symmetric=True),
        ])

        results = symm_pipeline.process(atm_data)

        assert processed_ids(results) == [0, 2]

    def test_no_symmetric_atms_raises_value_error(self, symm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=False),
            make_atm_trial(1, symmetric=False),
        ])

        with pytest.raises(ValueError, match="No symmetric ATMs"):
            symm_pipeline.process(atm_data)

    def test_non_contiguous_trial_ids_preserved(self, symm_pipeline):
        """
        process() iterates trial_indexes and looks trials up with get_trial_data(),
        so arbitrary IDs work correctly and are carried into the results.
        """
        atm_data = TrialATM([
            make_atm_trial(5, symmetric=True),
            make_atm_trial(7, symmetric=False),
            make_atm_trial(9, symmetric=True),
        ])

        results = symm_pipeline.process(atm_data)

        assert processed_ids(results) == [5, 9]

    def test_all_non_symmetric_trials_processed(self, nonsymm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=False),
            make_atm_trial(1, symmetric=False),
        ])

        results = nonsymm_pipeline.process(atm_data)

        assert processed_ids(results) == [0, 1]

    def test_symmetric_trials_are_skipped(self, nonsymm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=False),
            make_atm_trial(2, symmetric=False),
        ])

        results = nonsymm_pipeline.process(atm_data)

        assert processed_ids(results) == [1, 2]

    def test_labels_paired_with_surviving_trials(self, symm_pipeline):
        """
        Trial 1 is dropped, so the surviving labels must be 'rest' and 'other'.
        A filtering bug that built checked_labels from the unfiltered list
        would shift them and yield ['rest', 'MI'].
        """
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True, label="rest"),
            make_atm_trial(1, symmetric=False, label="task"),
            make_atm_trial(2, symmetric=True, label="other"),
        ])

        results = symm_pipeline.process(atm_data)

        assert processed_labels(results) == ["rest", "other"]

    def test_id_and_label_stay_together(self, symm_pipeline):
        """Each result must carry the id/label pair of its source trial."""
        atm_data = TrialATM([
            make_atm_trial(5, symmetric=True, label="rest"),
            make_atm_trial(7, symmetric=False, label="MI"),
            make_atm_trial(9, symmetric=True, label="other"),
        ])

        results = symm_pipeline.process(atm_data)

        assert [(t.trial_id, t.trial_label) for t in results.trials] == \
               [(5, "rest"), (9, "other")]

    def test_atm_method_copied_from_source_trial(self, symm_pipeline):
        atm_data = TrialATM([make_atm_trial(0, symmetric=True, method="standard")])

        results = symm_pipeline.process(atm_data)

        assert results.trials[0].atm_method == "standard"

    def test_all_symmetric_prints_confirmation(self, symm_pipeline, capsys):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=True),
        ])

        symm_pipeline.process(atm_data)

        assert "All ATMs are symmetric" in capsys.readouterr().out

    def test_all_non_symmetric_prints_confirmation(self, nonsymm_pipeline, capsys):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=False),
            make_atm_trial(1, symmetric=False),
        ])

        nonsymm_pipeline.process(atm_data)

        assert "All ATMs are non-symmetric" in capsys.readouterr().out

    def test_partial_match_prints_counts(self, symm_pipeline, capsys):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=False),
            make_atm_trial(2, symmetric=True),
        ])

        symm_pipeline.process(atm_data)

        assert "Found 2 symmetric ATMs on a total of 3" in capsys.readouterr().out

    def test_partial_match_prints_counts_non_symmetric(self, nonsymm_pipeline, capsys):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=False),
            make_atm_trial(2, symmetric=False),
        ])

        nonsymm_pipeline.process(atm_data)

        assert "Found 2 non-symmetric ATMs on a total of 3" in capsys.readouterr().out

    def test_no_non_symmetric_atms_raises_value_error(self, nonsymm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=True),
        ])

        with pytest.raises(ValueError, match="No non-symmetric ATMs found"):
            nonsymm_pipeline.process(atm_data)

    def test_returns_trial_metrics_object(self, symm_pipeline):
        atm_data = TrialATM([make_atm_trial(0, symmetric=True)])

        results = symm_pipeline.process(atm_data)

        assert isinstance(results, TrialMetrics)

    def test_one_result_per_selected_trial(self, symm_pipeline):
        atm_data = TrialATM([
            make_atm_trial(0, symmetric=True),
            make_atm_trial(1, symmetric=False),
            make_atm_trial(2, symmetric=True),
            make_atm_trial(3, symmetric=True),
        ])

        results = symm_pipeline.process(atm_data)

        assert len(results.trials) == 3

    @pytest.mark.parametrize("metric", list(METRIC_SPECS))
    def test_metric_calls_only_its_own_method(self, metric, atm_data_3symm):
        p = NetworkMetricsPipeline(metrics=[metric], symm=True)
        stub_all_metrics(p)

        p.process(atm_data_3symm) # 3 symmetric trials -> 3 calls expected

        expected_method = METRIC_SPECS[metric]["method"]
        for method_name in ALL_METHOD_NAMES:
            mock = getattr(p, method_name)
            expected_calls = 3 if method_name == expected_method else 0
            assert mock.call_count == expected_calls, (
                f"metric '{metric}': expected {expected_calls} call(s) to "
                f"{method_name}, got {mock.call_count}"
            )

    @pytest.mark.parametrize("metric", list(METRIC_SPECS))
    @patch(f"{MODULE}.MetricsObj")
    def test_metrics_sets_expected_attributes(self, mock_mo, metric):
        instance = MagicMock()
        mock_mo.return_value = instance      # MetricsObj(...) returns this

        atm_data = make_atm_data([make_atm_trial(0, symmetric=True)])
        p = NetworkMetricsPipeline(metrics=[metric], symm=True)
        stub_all_metrics(p)

        p.process(atm_data)

        for attr, expected in METRIC_SPECS[metric]["attrs"].items():
            actual = getattr(instance, attr)
            # assert_allclose handles both scalars and arrays
            np.testing.assert_allclose(
                actual, expected,
                err_msg=f"metric '{metric}': attribute '{attr}' is wrong",
            )

    @pytest.mark.parametrize("metric", list(METRIC_SPECS))
    def test_metric_receives_trial_mean_atm(self, metric):
        """Every metric must be handed the trial's mean_atm, not something else."""
        expected_matrix = symmetric_matrix()
        atm_data = make_atm_data([
            make_atm_trial(0, mean_atm=expected_matrix, symmetric=True)
        ])
        p = NetworkMetricsPipeline(metrics=[metric], symm=True)
        stub_all_metrics(p)

        p.process(atm_data)

        mock = getattr(p, METRIC_SPECS[metric]["method"])
        args, _ = mock.call_args
        np.testing.assert_array_equal(args[0], expected_matrix)

    def test_all_metrics_requested_computes_all(self, atm_data_3symm):
        p = NetworkMetricsPipeline(metrics=list(METRIC_SPECS), symm=True)
        stub_all_metrics(p)

        p.process(atm_data_3symm)

        for method_name in ALL_METHOD_NAMES:
            assert getattr(p, method_name).call_count == 3