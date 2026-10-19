import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")          # headless backend – no window, no display needed
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch

from pyAvalanches.dataset import Dataset, create_dataset_from_mne, plot_amplitude_distribution, _build_dataset
from pyAvalanches.avalanches import plot_avalanche_distribution, _find_avalanches, _binarize, _plot_distribution, _avalanche_stats

def make_data(n_trials=1, n_channels=3, n_times=20, activations=None):
    """
    Builds data with avalanches at exactly known positions.

    A small ramp is used as baseline instead of zeros, so no channel has zero
    variance (a flat channel would make z-scoring produce NaNs).

    Spikes are set to 100 - far above the baseline (max ~0.2) - so they remain
    clearly supra-threshold after z-scoring, whatever the exact cutoff rule.

    Parameters
    ----------
    activations : dict, optional
        {(trial, channel): [time indices]} - positions set to the spike value.
    """
    t = np.arange(n_times)
    ch = np.arange(n_channels)
    base = 0.01 * (t[None, None, :] + ch[None, :, None] + 1)
    data = np.repeat(base, n_trials, axis=0).astype(float)

    for (trial, channel), times in (activations or {}).items():
        data[trial, channel, times] = 100.0
    return data


def make_input_dict(data, trial_labels=None):
    """A dict in the form plot_avalanche_distribution() expects."""
    return {
        "data_array":    data,
        "samp_freq":     250.0,
        "fmin":          0.5,
        "fmax":          40.0,
        "channel_names": [f"ch{i}" for i in range(data.shape[1])],
        "trial_labels":  trial_labels,
        "t_max":         None,
    }


#: Single-trial layout with two hand-verifiable avalanches.
#:   avalanche 1 : t = 2,3,4  ->  size 4, duration 3
#:   avalanche 2 : t = 8,9    ->  size 2, duration 2
KNOWN_ACTIVATIONS = {
    (0, 0): [2, 3, 4],
    (0, 1): [3],
    (0, 2): [8, 9],
}
KNOWN_SIZES     = [4, 2]
KNOWN_DURATIONS = [3, 2]


#: Default arguments, so each test only states what it cares about.
DEFAULTS = dict(
    plot="size",
    threshold_method="fixed",
    threshold=2.0,
    time_bins=[1],
    hist_bins=10,
    linear=0,
)


def call(data, **overrides):
    """Calls the function with DEFAULTS, overridden by any kwargs given."""
    return plot_avalanche_distribution(data, **{**DEFAULTS, **overrides})

@pytest.fixture(autouse=True)
def close_figures():
    """Runs after every test in this module."""
    yield
    plt.close("all")


@pytest.fixture
def known_dict():
    """One trial, two known avalanches, no condition labels."""
    return make_input_dict(make_data(activations=KNOWN_ACTIVATIONS))


@pytest.fixture
def two_condition_dict():
    """
    3 trials labelled rest / MI / rest, each with its own avalanche:
      trial 0 (rest) : t 2-4    -> duration 3
      trial 1 (MI)   : t 5-6    -> duration 2
      trial 2 (rest) : t 10-13  -> duration 4
    """
    data = make_data(n_trials=3, activations={
        (0, 0): [2, 3, 4],
        (1, 0): [5, 6],
        (2, 0): [10, 11, 12, 13],
    })
    return make_input_dict(data, trial_labels=["rest", "MI", "rest"])


@pytest.fixture
def noisy_dict():
    """Random noise - produces many avalanches of varied size."""
    data = np.random.default_rng(0).standard_normal((1, 8, 2000))
    return make_input_dict(data)


class TestInputTypes:

    def test_dict_input_accepted(self, known_dict):
        fig, _, _ = call(known_dict)

        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize("bad_input", [
        np.zeros((1, 3, 20)), [1, 2, 3], "file.fif", 42, None,
    ])
    def test_unsupported_type_raises_type_error(self, bad_input):
        with pytest.raises(TypeError, match="must be an mne Raw"):
            call(bad_input)

    def test_missing_dict_key_raises_key_error(self, known_dict):
        del known_dict["samp_freq"]

        with pytest.raises(KeyError, match="samp_freq"):
            call(known_dict)

    def test_all_missing_keys_reported_together(self, known_dict):
        """The error lists every missing key, not just the first."""
        del known_dict["fmin"]
        del known_dict["fmax"]

        with pytest.raises(KeyError, match="fmax.*fmin"):
            call(known_dict)

    def test_mne_raw_accepted(self):
        mne = pytest.importorskip("mne")

        data = make_data(activations=KNOWN_ACTIVATIONS)[0]   # (3, 20)
        info = mne.create_info(["ch0", "ch1", "ch2"], sfreq=250.0, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)

        fig, _, results = call(raw)

        assert isinstance(fig, plt.Figure)
        assert list(results["global"][1]) == KNOWN_SIZES

    def test_mne_epochs_array_accepted(self):
        """
        EpochsArray subclasses BaseEpochs but not Epochs, so this only works
        because the isinstance check uses BaseEpochs.
        """
        mne = pytest.importorskip("mne")

        data = make_data(n_trials=2, activations={(0, 0): [2, 3, 4],
                                                  (1, 0): [5, 6]})
        info = mne.create_info(["ch0", "ch1", "ch2"], sfreq=250.0, ch_types="eeg")
        epochs = mne.EpochsArray(data, info, verbose=False)

        fig, _, _ = call(epochs)

        assert isinstance(fig, plt.Figure)

class TestArgumentValidation:

    @pytest.mark.parametrize("bad_plot", ["Size", "sizes", "", None])
    def test_invalid_plot_raises(self, known_dict, bad_plot):
        with pytest.raises(ValueError, match="plot must be one of"):
            call(known_dict, plot=bad_plot)

    @pytest.mark.parametrize("bad_method", ["Fixed", "STD", "", None])
    def test_invalid_threshold_method_raises(self, known_dict, bad_method):
        with pytest.raises(ValueError, match="threshold_method must be one of"):
            call(known_dict, threshold_method=bad_method)

    @pytest.mark.parametrize("bad_linear", [-1, 2, 0.5, None])
    def test_invalid_linear_raises(self, known_dict, bad_linear):
        with pytest.raises(ValueError, match="linear must be either 0 or 1"):
            call(known_dict, linear=bad_linear)

    @pytest.mark.parametrize("bad_bins", [[], [0], [-1], [1, 0]])
    def test_invalid_time_bins_raises(self, known_dict, bad_bins):
        with pytest.raises(ValueError, match="time_bins"):
            call(known_dict, time_bins=bad_bins)

    def test_invalid_hist_bins_raises(self, known_dict):
        with pytest.raises(ValueError, match="hist_bins must be >= 1"):
            call(known_dict, hist_bins=0)

    @pytest.mark.parametrize("plot", ["size", "duration"])
    def test_valid_plot_options_accepted(self, known_dict, plot):
        fig, _, _ = call(known_dict, plot=plot)

        assert isinstance(fig, plt.Figure)

    @pytest.mark.parametrize("linear", [0, 1])
    def test_valid_linear_options_accepted(self, known_dict, linear):
        fig, _, _ = call(known_dict, linear=linear)

        assert isinstance(fig, plt.Figure)

    def test_validation_runs_before_input_conversion(self):
        """
        Arguments are checked first, so a bad `plot` is reported even when
        the input type is also invalid.
        """
        with pytest.raises(ValueError, match="plot must be one of"):
            call("not_valid_input", plot="nonsense")

    def test_no_activations_raises(self, known_dict):
        """A threshold above every sample leaves nothing to analyse."""
        with pytest.raises(ValueError, match="No activations above threshold"):
            call(known_dict, threshold=1000.0)


class TestReturnStructure:

    def test_always_returns_three_values(self, known_dict):
        result = call(known_dict)

        assert len(result) == 3

    def test_fig_cond_is_none_for_single_condition(self, known_dict):
        fig, fig_cond, results = call(known_dict)

        assert isinstance(fig, plt.Figure)
        assert fig_cond is None
        assert isinstance(results, dict)

    def test_fig_cond_is_a_figure_for_multiple_conditions(self, two_condition_dict):
        _, fig_cond, _ = call(two_condition_dict)

        assert isinstance(fig_cond, plt.Figure)

    def test_global_results_keyed_by_time_bin(self, known_dict):
        _, _, results = call(known_dict, time_bins=[1, 2, 4])

        assert set(results["global"]) == {1, 2, 4}

    def test_by_condition_absent_for_single_condition(self, known_dict):
        _, _, results = call(known_dict)

        assert "by_condition" not in results

    def test_by_condition_present_for_multiple_conditions(self, two_condition_dict):
        _, _, results = call(two_condition_dict)

        assert set(results["by_condition"]) == {"rest", "MI"}

    def test_global_results_kept_alongside_conditions(self, two_condition_dict):
        """Both keys survive - the global results are not overwritten."""
        _, _, results = call(two_condition_dict)

        assert "global" in results
        assert "by_condition" in results


class TestAvalancheValues:
    """
    Ground truth for `known_dict`:

        ch0  active at t = 2, 3, 4
        ch1  active at t = 3
        ch2  active at t = 8, 9

      avalanche 1 : t 2-4, activations [3, 1, 0] -> size 4, duration 3
      avalanche 2 : t 8-9, activations [0, 0, 2] -> size 2, duration 2
    """

    def test_sizes_match_expected(self, known_dict):
        _, _, results = call(known_dict, plot="size")

        assert list(results["global"][1]) == KNOWN_SIZES

    def test_durations_match_expected(self, known_dict):
        _, _, results = call(known_dict, plot="duration")

        assert list(results["global"][1]) == KNOWN_DURATIONS

    def test_avalanche_count(self, known_dict):
        _, _, results = call(known_dict)

        assert len(results["global"][1]) == 2

    def test_both_threshold_methods_find_the_same_avalanches(self, known_dict):
        """The spikes are far above baseline, so both cutoffs isolate them."""
        _, _, fixed = call(known_dict, threshold_method="fixed", threshold=2.0)
        _, _, std   = call(known_dict, threshold_method="std",   threshold=2.0)

        assert list(fixed["global"][1]) == list(std["global"][1])

    def test_results_are_lists(self, known_dict):
        """Sizes and durations both come back as arrays, consistently."""
        _, _, size_res = call(known_dict, plot="size")
        _, _, dur_res  = call(known_dict, plot="duration")

        assert isinstance(size_res["global"][1], list)
        assert isinstance(dur_res["global"][1], list)

    def test_results_are_scale_invariant(self):
        """
        Data is z-scored before thresholding, so multiplying every value by a
        constant must leave the detected avalanches unchanged.
        """
        base = make_data(activations=KNOWN_ACTIVATIONS)

        _, _, small = call(make_input_dict(base * 1e-6))
        _, _, large = call(make_input_dict(base * 1e6))

        assert list(small["global"][1]) == list(large["global"][1])

    def test_trials_are_concatenated_along_time(self):
        """
        Trials are laid end to end, so activity at the last sample of one
        trial and the first sample of the next merges into ONE avalanche.
        """
        data = make_data(n_trials=2, n_times=20, activations={
            (0, 0): [19],
            (1, 0): [0],
        })

        _, _, results = call(make_input_dict(data), plot="duration")

        assert list(results["global"][1]) == [2]


class TestTimeBins:

    def test_one_entry_per_requested_bin_size(self, known_dict):
        _, _, results = call(known_dict, time_bins=[1, 2, 5])

        assert list(results["global"]) == [1, 2, 5]

    def test_merging_preserves_total_size(self, known_dict):
        """Subsampling sums activations, so no activation is lost."""
        _, _, k1 = call(known_dict, plot="size", time_bins=[1])
        _, _, k4 = call(known_dict, plot="size", time_bins=[4])

        assert sum(k1["global"][1]) == sum(k4["global"][4])

    def test_remainder_samples_are_dropped(self):
        """
        Trailing samples that do not fill a complete bin are discarded, so
        an avalanche living there disappears (21 % 2 == 1).
        """
        data = make_data(n_times=21, activations={
            (0, 0): [2, 3],      # kept
            (0, 1): [20],        # in the dropped remainder
        })

        _, _, results = call(make_input_dict(data), time_bins=[2])

        assert len(results["global"][2]) == 1


class TestConditions:

    def test_each_condition_uses_all_its_trials(self, two_condition_dict):
        """
        'rest' covers trials 0 and 2 (durations 3 and 4), 'MI' covers trial 1
        (duration 2). A faulty trial-selection loop would drop trial 2.
        """
        _, _, results = call(two_condition_dict, plot="duration")

        assert sorted(results["by_condition"]["rest"][1]) == [3, 4]
        assert list(results["by_condition"]["MI"][1])     == [2]

    def test_condition_not_at_the_start_is_handled(self, two_condition_dict):
        """'MI' is at index 1, not index 0 - it must still be found."""
        _, _, results = call(two_condition_dict)

        assert len(results["by_condition"]["MI"][1]) == 1

    def test_one_subplot_per_condition(self, two_condition_dict):
        _, fig_cond, _ = call(two_condition_dict)

        assert len(fig_cond.axes) == 2

    def test_condition_order_is_deterministic(self, two_condition_dict):
        """Conditions are sorted, so the subplot order is reproducible."""
        _, fig_cond, _ = call(two_condition_dict)

        titles = [ax.get_title() for ax in fig_cond.axes]
        assert titles == ["Condition MI", "Condition rest"]

    def test_single_repeated_label_counts_as_one_condition(self):
        """All trials sharing one label means no per-condition figure."""
        data = make_data(n_trials=2, activations={(0, 0): [2, 3, 4],
                                                  (1, 0): [5, 6]})
        d = make_input_dict(data, trial_labels=["rest", "rest"])

        _, fig_cond, results = call(d)

        assert fig_cond is None
        assert "by_condition" not in results

    def test_condition_results_keyed_by_time_bin(self, two_condition_dict):
        _, _, results = call(two_condition_dict, time_bins=[1, 2])

        assert set(results["by_condition"]["rest"]) == {1, 2}

class TestFigureStructure:

    def test_global_figure_has_one_axes(self, known_dict):
        fig, _, _ = call(known_dict)

        assert len(fig.axes) == 1

    def test_one_curve_per_time_bin(self, noisy_dict):
        time_bins = [1, 2, 4]
        fig, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0,
                         time_bins=time_bins)

        ax = fig.axes[0]
        assert len(ax.collections) == len(time_bins)   # scatter markers
        assert len(ax.lines)       == len(time_bins)   # connecting lines

    def test_both_axes_are_logarithmic(self, known_dict):
        fig, _, _ = call(known_dict)

        assert fig.axes[0].get_xscale() == "log"
        assert fig.axes[0].get_yscale() == "log"

    @pytest.mark.parametrize("plot, xlabel, title", [
        ("size",     "Avalanche Size",     "Avalanche Size Distribution"),
        ("duration", "Avalanche Duration", "Avalanche Duration Distribution"),
    ])
    def test_labels_match_plotted_quantity(self, known_dict, plot, xlabel, title):
        fig, _, _ = call(known_dict, plot=plot)

        assert fig.axes[0].get_xlabel() == xlabel
        assert fig.axes[0].get_title()  == title

    def test_y_label_is_probability_density(self, known_dict):
        fig, _, _ = call(known_dict)

        assert fig.axes[0].get_ylabel() == "Probability Density"

    def test_legend_names_each_time_bin(self, noisy_dict):
        fig, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0,
                         time_bins=[1, 4])

        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["time bins = 1", "time bins = 4"]

    def test_plotted_values_are_positive_and_finite(self, noisy_dict):
        """Log axes cannot show zero or negative densities."""
        fig, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0)

        y = fig.axes[0].lines[0].get_ydata()
        assert len(y) > 0
        assert np.all(y > 0)
        assert np.all(np.isfinite(y))

    def test_plotted_points_never_exceed_hist_bins(self, noisy_dict):
        fig, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0,
                         hist_bins=15)

        assert len(fig.axes[0].lines[0].get_xdata()) <= 15

class TestBinning:

    def test_identical_values_give_one_bin(self):
        """
        When every avalanche has the same size, xmax == xmin - a degenerate
        case that must return a single bin rather than raising.
        """
        data = make_data(n_times=20, activations={
            (0, 0): [2], (0, 1): [8], (0, 2): [14],
        })

        fig, _, results = call(make_input_dict(data), plot="size")

        assert list(results["global"][1]) == [1, 1, 1]
        assert len(fig.axes[0].lines[0].get_xdata()) == 1

    def test_log_and_linear_give_different_bin_centers(self, noisy_dict):
        fig_log, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0,
                             linear=0)
        fig_lin, _, _ = call(noisy_dict, threshold_method="std", threshold=2.0,
                             linear=1)

        x_log = fig_log.axes[0].lines[0].get_xdata()
        x_lin = fig_lin.axes[0].lines[0].get_xdata()

        assert not np.array_equal(x_log, x_lin)

    def test_plotted_x_values_lie_within_the_data_range(self, noisy_dict):
        fig, _, results = call(noisy_dict, threshold_method="std", threshold=2.0)

        x = fig.axes[0].lines[0].get_xdata()
        values = results["global"][1]
        assert min(x) >= min(values)
        assert max(x) <= max(values)

