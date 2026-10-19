import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pyAvalanches.avalanches import AvalanchePipeline  # adjust to your actual import

@pytest.fixture
def fixed_pipeline():
    """Fixed-threshold pipeline – the most common configuration in tests."""
    return AvalanchePipeline(
        z_scoring = 'by_trial',
        #z_score_by_trial=False,
        threshold_method="fixed",
        threshold=2.0,
    )

@pytest.fixture
def std_pipeline():
    """Std-based pipeline, same threshold multiplier."""
    return AvalanchePipeline(
        z_scoring = 'by_trial',
        threshold_method="std",
        threshold=2.0,
    )

@pytest.fixture
def bin_pipeline():
    """
    Pipeline suited for testing _compute_avalanches with already-binarized
    (0/1) data; threshold=0.5 cleanly separates the two values.
    """
    return AvalanchePipeline(
        z_scoring = 'by_trial',
        threshold_method="fixed",
        threshold=0.5,
        min_duration=1,
        min_size=1,
        bins=1,
    )

@pytest.fixture
def simple_bin_data():
    """
    3 channels × 7 time-bins.  A single contiguous avalanche sits at
    columns 2–4 (zero-indexed).

    ch\\t  0  1  2  3  4  5  6
    ──────────────────────────
     0    0  0  1  1  1  0  0
     1    0  0  1  1  0  0  0
     2    0  0  0  1  1  0  0

    Ground truth: bounds=(2, 4), duration=3, size=7.
    """
    data = np.zeros((3, 7), dtype=int)
    data[0, 2:5] = 1
    data[1, 2:4] = 1
    data[2, 3:5] = 1
    return data

@pytest.fixture
def two_aval_bin_data():
    """
    3 channels × 7 time-bins, two avalanches

    ch\\t  0  1  2  3  4  5  6
    ──────────────────────────
     0    1  0  0  0  0  0  0
     1    1  1  0  1  0  1  0
     2    0  0  0  1  1  0  0

    Ground truth: bounds=(2, 4), duration=3, size=7.
    """
    data = np.zeros((3, 7), dtype=int)
    data[0, 0] = 1
    data[1, 0:2] = 1
    data[1, 3] = 1
    data[1, 5] = 1
    data[2, 3:5] = 1
    return data

def make_mock_dataset(n_trials=3, trial_labels = None, n_channels=3, n_time=50):
    """
    Returns a MagicMock configured to behave like a Dataset instance.
    """

    np.random.seed(42)

    ds = MagicMock()
    ds.trial_ids = list(range(n_trials))
    ds.trial_labels = trial_labels
    ds.data = {i: np.random.randn(n_channels, n_time) for i in range(n_trials)}

    return ds

def _avalanche_output(n_channels=3):
    """
    A valid 8-tuple that mirrors the real _compute_avalanches return value.
    Assigned to _compute_avalanches as its return_value so that the rest
    of process()/transform() receives well-typed data and does not crash.
    """
    return (
        [np.zeros((n_channels, 5))],    # matrices
        [5],                             # sizes
        [3],                             # durations
        [np.ones(n_channels)],           # patterns
        [np.zeros(n_channels)],          # patterns_temporal
        [(2, 4)],                        # avalanche_bounds
        1,                               # flexibility
        1,                               # avalanche_count
    )

@pytest.fixture
def array_3d():
    """Shape: (3 trials, 3 channels, 50 time points)."""
    np.random.seed(42)
    return np.random.randn(3, 3, 50)

@pytest.fixture
def valid_info_dict():
    """Minimal info dict required when passing a np.ndarray to process()."""
    np.random.seed(42)
    return {
        "data_array":    np.random.randn(3, 3, 50),
        "samp_freq":     250.0,
        "fmin":          8.0,
        "fmax":          35.0,
        "channel_names": ["ch0", "ch1", "ch2"],
        "trial_labels":  [0, 1, 0],
        "t_max":         None
    }

def process_avalanches(pipeline, bin_data: np.ndarray) -> dict:
    (matrices, sizes, durations,
     patterns, patterns_temp,
     bounds, flexibility, avalanche_count) = pipeline._compute_avalanches(bin_data)
    return dict(
        matrices=matrices,
        sizes=sizes,
        durations=durations,
        patterns=patterns,
        patterns_temp=patterns_temp,
        bounds=bounds,
        flexibility=flexibility,
        avalanche_count=avalanche_count,
    )

# =============================================================================
# 1 · __init__
# Rule of thumb: test the happy path, the error path, and default values.
# =============================================================================

class TestInit:

    # testo gli input di threshold perché se sono sbagliati devono restituire un errore
    @pytest.mark.parametrize("correct_method", ["fixed", "std"])
    def test_valid_threshold_methods(self, correct_method):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method=correct_method,
            threshold=1.0
        )
        assert p.threshold_method == correct_method

    @pytest.mark.parametrize("wrong_method", ["Fixed", "STD", "zscore", "sd", "", 42, None])
    def test_invalid_threshold_methods(self, wrong_method):
        with pytest.raises(ValueError, match="method must be one of"):
            AvalanchePipeline(
                z_scoring = 'by_trial',
                threshold_method=wrong_method,
                threshold=1.0
            )

    # testo se i parametri opzionali sono inizializzati correttamente
    def test_optional_parameters_initialization(self):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="fixed",
            threshold=1.0
        )
        assert p.min_duration == 1
        assert p.min_size == 1
        assert p.bins == 1
        assert p.trial_label is None

    # testo se tutti gli argomenti vengono inizialittati come attributi
    def test_all_arguments_stored_as_attributes(self):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="std",
            threshold=3.5,
            min_duration=2,
            min_size=4,
            bins=5,
            trial_label=1,
        )
        assert p.z_scoring == 'by_trial'
        assert p.threshold_method == "std"
        assert p.threshold == 3.5
        assert p.min_duration == 2
        assert p.min_size == 4
        assert p.bins == 5
        assert p.trial_label == 1

class TestThresholdDataFixed:

    def test_value_above_threshold_becomes_one(self, fixed_pipeline):
        data = np.array([[3.0, 0.5],
                         [0.1, 5.0]])
        result = fixed_pipeline._threshold_data_fixed(data)
        assert result[0, 0] == 1  # 3.0 > 2.0 → 1
        assert result[1, 1] == 1  # 5.0 > 2.0 → 1

    def test_value_below_threshold_become_zero(self, fixed_pipeline):
        data = np.array([[3.0, 0.5],
                         [0.1, 5.0]])
        result = fixed_pipeline._threshold_data_fixed(data)
        assert result[0, 1] == 0
        assert result[1, 0] == 0

    def test_negative_values_compared_by_absolute_value(self, fixed_pipeline):
        data = np.array([[-3.0, -0.5],
                         [0.1, 5.0]])
        result = fixed_pipeline._threshold_data_fixed(data)
        assert result[0, 0] == 1
        assert result[0, 1] == 0

    def test_output_shape_matches_input_shape(self, fixed_pipeline):
        data = np.random.randn(8,150)
        assert fixed_pipeline._threshold_data_fixed(data).shape == data.shape

    def test_output_contains_only_zeros_and_ones(self, fixed_pipeline):
        data = np.random.randn(8,150)
        unique_values = set(np.unique(fixed_pipeline._threshold_data_fixed(data)))
        assert unique_values.issubset({0,1})

    def test_warning_no_activations(self, fixed_pipeline, capsys):
        silent_data = np.full((3, 20), 1.0) # all < 2.0
        fixed_pipeline._threshold_data_fixed(silent_data)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestThresholdDataStd:

    def test_output_shape_matches_input_shape(self, std_pipeline):
        mean = 1
        std = 1
        data = np.random.randn(8,150)
        assert std_pipeline._threshold_data_std(data, mean, std).shape == data.shape

    def test_threshold_matches_manual_formula(self, std_pipeline):
        """
        Replicate the formula ourselves and compare element-wise.
        This pattern ("compute expected, compare to actual") is the
        most direct way to verify algorithmic correctness.
        """
        np.random.seed(42)
        mean = 1
        std = 1
        data = np.random.randn(3, 50)
        result = std_pipeline._threshold_data_std(data, mean, std)
        expected = np.where(np.abs(data) > (mean + (2.0 * std)), 1, 0)
        #expected = np.where(np.abs(data) > 2.0 * data.std(), 1, 0)
        np.testing.assert_array_equal(result, expected)

    def test_output_contains_only_zeros_and_ones(self, std_pipeline):
        mean = 1
        std = 1
        data = np.random.randn(8,150)
        unique_values = set(np.unique(std_pipeline._threshold_data_std(data, mean, std)))
        assert unique_values.issubset({0,1})

    def test_warning_no_activations(self, std_pipeline, capsys):
        mean = 1
        std = 1
        silent_data = np.full((3, 20), 0.0) # all < 2.0
        std_pipeline._threshold_data_std(silent_data, mean, std)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out


class TestComputeAvalanches:

    def test_all_zero_warning(self, bin_pipeline, capsys):
        zero_data = np.zeros((8,15))
        process_avalanches(bin_pipeline, zero_data)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_all_zeros_returns_empty_containers(self, bin_pipeline):
        zero_data = np.zeros((8, 15))
        results = process_avalanches(bin_pipeline, zero_data)
        assert results["avalanche_count"] == 0
        assert results["flexibility"] == 0
        for key in ("matrices", "sizes", "durations",
                    "patterns", "patterns_temp", "bounds"):
            assert results[key] == []

    def test_one_block_one_avalanche(self, bin_pipeline, simple_bin_data):
        results = process_avalanches(bin_pipeline, simple_bin_data)
        assert results["avalanche_count"] == 1

    def test_two_blocks_two_avalanches(self, bin_pipeline, two_aval_bin_data):
        results = process_avalanches(bin_pipeline, two_aval_bin_data)
        assert results["avalanche_count"] == 2

    def test_bounds(self, bin_pipeline, simple_bin_data):
        results = process_avalanches(bin_pipeline, simple_bin_data)
        assert results["bounds"][0] == (2,4)

    def test_size(self, bin_pipeline, simple_bin_data):
        results = process_avalanches(bin_pipeline, simple_bin_data)
        assert results["sizes"][0] == 7

    def test_duration(self, bin_pipeline, simple_bin_data):
        results = process_avalanches(bin_pipeline, simple_bin_data)
        assert results["durations"][0] == 3

    @pytest.mark.parametrize("min_dur, expected_count", [
        (4,0), # below minimum all excluded
        (3,1), # same as minimum, one passes
        (1,2) # above minimum, all kept
    ])
    def test_min_duration(self, min_dur, expected_count, two_aval_bin_data):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="fixed",
            threshold=0.5,
            min_duration=min_dur,
            min_size=1
        )
        results = process_avalanches(p, two_aval_bin_data)
        assert results["avalanche_count"] == expected_count

    @pytest.mark.parametrize("min_size, expected_count", [
        (5, 0),  # below minimum all excluded
        (4, 1),  # same as minimum, one passes
        (1, 2)  # above minimum, all kept
    ])
    def test_min_size(self, min_size, expected_count, two_aval_bin_data):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="fixed",
            threshold=0.5,
            min_duration=1,
            min_size=min_size
        )
        results = process_avalanches(p, two_aval_bin_data)
        assert results["avalanche_count"] == expected_count

    def test_spatial_pattern(self, bin_pipeline):
        data = np.zeros((3, 7), dtype=int)
        data[0, 0] = 1
        data[1, 0:2] = 1
        data[1, 3] = 1
        data[1, 5] = 1
        data[2, 4] = 1

        patterns = process_avalanches(bin_pipeline, data)["patterns"]
        np.testing.assert_array_equal(patterns[0], [1, 1, 0])
        np.testing.assert_array_equal(patterns[1], [0, 1, 1])

    def test_temporal_pattern(self, bin_pipeline):
        data = np.zeros((3, 7), dtype=int)
        data[0, 0] = 1
        data[1, 0:2] = 1
        data[1, 3] = 1
        data[1, 5] = 1
        data[2, 4] = 1

        patterns = process_avalanches(bin_pipeline, data)["patterns_temp"]
        np.testing.assert_array_equal(patterns[0], [1, 1, 0])
        np.testing.assert_array_equal(patterns[1], [0, 1, 2])

    def test_flexibility(self, bin_pipeline):
        data = np.zeros((3, 7), dtype=int)
        data[0, 0] = 1
        data[1, 0:2] = 1
        data[1, 3] = 1
        data[1, 5] = 1
        data[2, 4] = 1

        results = process_avalanches(bin_pipeline, data)
        assert results["flexibility"] == 2

    def test_subsampling(self, two_aval_bin_data):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="fixed",
            threshold=0.5,
            min_duration=1,
            min_size=1,
            bins=2
        )

        results = process_avalanches(p, two_aval_bin_data)
        assert results["avalanche_count"] == 1
        assert results["durations"][0] == 3
        assert results["sizes"][0] == 7
        np.testing.assert_array_equal(results["patterns"][0], [1, 1, 1])

    def test_extracted_matrix_shape_is_channels_by_duration(self, bin_pipeline, simple_bin_data):
        """Each matrix in 'matrices' must be (n_channels, avalanche_duration)."""
        n_channels = 3
        duration = 3
        matrices = process_avalanches(bin_pipeline, simple_bin_data)["matrices"]
        assert matrices[0].shape == (n_channels, duration)

class TestProcess:

    @pytest.mark.parametrize("wrong_input",[
        [1, 2, 3],  # list
        42,  # int
        "signal",  # str
        3.14,  # float
    ])
    def test_error_unsupported_type(self, fixed_pipeline, wrong_input):
        with pytest.raises(TypeError, match="input_data must be"):
            fixed_pipeline.process(wrong_input)

    def test_absent_label_raises_value_error(self, array_3d, valid_info_dict):
        p = AvalanchePipeline(
            z_scoring = 'by_trial', threshold_method="fixed",
            threshold=0.5, trial_label=99,  # does not exist in [0,1,0]
        )

        with pytest.raises(ValueError, match="No trials found with label"):
            p.process(valid_info_dict)

    @pytest.mark.parametrize("label, label_list, count, index", [
        (None, [0,1,0], 3, [0,1,2]), # no label
        (1, [1], 1, [1]),            # valid label for one trial
        (0, [0,0], 2, [0,2])         # valid label for two trials
    ])
    def test_label_selection(self, array_3d, valid_info_dict,
                             label, label_list, count, index):
        p = AvalanchePipeline(
            z_scoring = 'by_trial', threshold_method="fixed",
            threshold=0.5, trial_label=label,  # does not exist in [0,1,0]
        )
        results = p.process(valid_info_dict)
        assert results.trial_labels == label_list
        assert results.trial_count == count
        assert results.trial_indexes == index

    def test_all_trials_are_processed_correctly(self, array_3d, valid_info_dict, fixed_pipeline):
        results = fixed_pipeline.process(valid_info_dict)
        aval_count_trial_0 = results.trials[0].avalanche_count
        assert results.trial_count == 3            # number of trials
        assert results.trial_labels == [0,1,0]     # labels
        assert results.trial_indexes == [0,1,2]    # indexes
        assert len(results.trials) == 3            # number of AvalancheObj created
        assert len(results.trials[0].avalanches_mat) == aval_count_trial_0
        assert len(results.trials[0].durations) == aval_count_trial_0
        assert len(results.trials[0].bounds) == aval_count_trial_0
        assert len(results.trials[0].sizes) == aval_count_trial_0

class TestFit:

    def test_returns_self(self, fixed_pipeline, array_3d):
        """
        sklearn.Pipeline chains calls on the return value of fit(), so
        returning anything other than self would silently break it.
        This is the ONE contract that matters for a stub fit().
        """
        assert fixed_pipeline.fit(array_3d) is fixed_pipeline

    def test_does_not_mutate_any_attribute(self, fixed_pipeline, array_3d):
        """
        A stub must have no side-effects.  We snapshot __dict__ before
        and compare after to detect any accidental state change.
        """
        snapshot = fixed_pipeline.__dict__.copy()
        fixed_pipeline.fit(array_3d)
        assert fixed_pipeline.__dict__ == snapshot

    def test_accepts_optional_y_argument(self, fixed_pipeline, array_3d):
        """
        sklearn convention: fit(X, y=None) must tolerate a y argument
        without raising, even if it ignores it.
        """
        assert fixed_pipeline.fit(array_3d, y=np.zeros(3)) is fixed_pipeline

class TestTransform: # simplified version of process

    @pytest.mark.parametrize("wrong_input", [
        {"key": "value"},  # dict
        [1, 2, 3],  # list
        42,  # int
        "signal",  # str
        3.14,  # float
    ])
    def test_error_unsupported_type(self, fixed_pipeline, wrong_input):
        with pytest.raises(TypeError, match="X must be a numpy array"):
            fixed_pipeline.transform(wrong_input)

    def test_all_trials_are_processed_correctly(self, array_3d, std_pipeline):
        p = AvalanchePipeline(
            z_scoring = 'by_trial',
            threshold_method="std",
            threshold=0.5
        )
        results = p.transform(array_3d)
        aval_count_trial_0 = results.trials[0].avalanche_count
        assert results.trial_count == 3            # number of trials
        assert results.trial_indexes == [0,1,2]    # indexes
        assert len(results.trials) == 3            # number of AvalancheObj created
        assert len(results.trials[0].avalanches_mat) == aval_count_trial_0
        assert len(results.trials[0].durations) == aval_count_trial_0
        assert len(results.trials[0].bounds) == aval_count_trial_0
        assert len(results.trials[0].sizes) == aval_count_trial_0