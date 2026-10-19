import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pyAvalanches.dataset import Dataset, create_dataset_from_mne, _build_dataset

MODULE = "pyAvalanches.dataset"

def make_data_3d(n_trials=3, n_channels=4, n_times=100, seed=42):
    """Deterministic 3-D data array, shape (n_trials, n_channels, n_time_points)."""
    return np.random.default_rng(seed).standard_normal((n_trials, n_channels, n_times))


def make_dataset(n_trials=3, n_channels=4, n_times=100,
                 samp_freq=250.0, fmin=0.5, fmax=40.0,
                 channel_names=None, trial_labels=None, t_max=None, info=None,
                 data=None):
    """
    A real Dataset. Defaults give a valid 3-D dataset with 3 trials.
    trial_labels defaults to None (not auto-filled) so tests can opt in.
    """
    if data is None:
        data = make_data_3d(n_trials, n_channels, n_times)
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    return Dataset(
        data=data,
        samp_freq=samp_freq,
        fmin=fmin,
        fmax=fmax,
        channel_names=channel_names,
        trial_labels=trial_labels,
        t_max=t_max,
        info=info,
    )


@pytest.fixture
def dataset():
    """3 trials x 4 channels x 100 time points, no trial labels."""
    return make_dataset()


@pytest.fixture
def labelled_dataset():
    """Same shape, with one label per trial."""
    return make_dataset(trial_labels=["rest", "task", "rest"])


@pytest.fixture
def dataset_factory():
    return make_dataset

class TestInit:

    def test_required_arguments_stored(self, dataset):
        assert dataset.samp_freq == 250.0
        assert dataset.fmin      == 0.5
        assert dataset.fmax      == 40.0
        assert dataset.data.shape == (3, 4, 100)

    def test_optional_arguments_default_to_none(self):
        ds = Dataset(data=make_data_3d(), samp_freq=250.0, fmin=0.5, fmax=40.0)

        assert ds.channel_names is None
        assert ds.trial_labels is None
        assert ds.t_max is None
        assert ds.data_zscored is None
        assert ds.info is None

    def test_channel_names_stored(self, dataset):
        assert dataset.channel_names == ["ch0", "ch1", "ch2", "ch3"]

    def test_trial_labels_stored(self, labelled_dataset):
        assert labelled_dataset.trial_labels == ["rest", "task", "rest"]

    def test_info_dict_stored(self):
        ds = make_dataset(info={"bads": ["ch1"]})
        assert ds.info == {"bads": ["ch1"]}

    def test_trial_ids_generated_from_first_axis(self, dataset):
        """trial_ids is always auto-generated as 0..n_trials-1."""
        assert dataset.trial_ids == [0, 1, 2]

    def test_data_is_copied(self):
        """
        self.data = data stores a copy, so mutating the caller's array
        doesn't change the Dataset.
        """
        arr = make_data_3d()
        ds = make_dataset(data=arr)

        arr[0, 0, 0] = 999.0

        assert ds.data[0, 0, 0] != 999.0

    def test_t_max_should_truncate(self, dataset_factory):
        ds = dataset_factory(n_times=100, t_max=50)
        assert ds.data.shape[-1] == 50

    def test_invalid_t_max_should_raise(self, dataset_factory):
        with pytest.raises(ValueError, match="must be a positive"):
            dataset_factory(t_max=0)

    def test_oversized_t_max_should_raise(self, dataset_factory):
        with pytest.raises(ValueError, match="exceeds the available"):
            dataset_factory(n_times=100, t_max=5000)

    def test_mismatched_labels_should_raise(self, dataset_factory):
        with pytest.raises(ValueError, match="trial_labels length must match"):
            dataset_factory(n_trials=3, trial_labels=["rest", "task"])

    def test_2d_input_should_become_single_trial(self):
        ds = make_dataset(data=np.zeros((4, 100)))
        assert ds.data.shape == (1, 4, 100)
        assert ds.n_trials == 1

    def test_1d_input_should_raise(self):
        with pytest.raises(ValueError, match="must be 2D .* or 3D"):
            make_dataset(data=np.zeros(100))

class TestShapeProperties:

    def test_n_trials(self, dataset):
        assert dataset.n_trials == 3

    def test_n_channels(self, dataset):
        assert dataset.n_channels == 4

    def test_n_time_points(self, dataset):
        assert dataset.n_time_points == 100

    @pytest.mark.parametrize("n_trials, n_channels, n_times", [
        (1, 2, 10),
        (5, 64, 500),
        (10, 3, 1000),
    ])
    def test_properties_match_data_shape(self, dataset_factory,
                                         n_trials, n_channels, n_times):
        ds = dataset_factory(n_trials=n_trials, n_channels=n_channels,
                             n_times=n_times)

        assert (ds.n_trials, ds.n_channels, ds.n_time_points) == \
               (n_trials, n_channels, n_times)

    def test_channel_names_length_matches_n_channels(self, dataset):
        assert len(dataset.channel_names) == dataset.n_channels

    def test_mismatched_channels_should_raise(self, dataset_factory):
        with pytest.raises(ValueError, match="channel names length must match"):
            dataset_factory(n_trials=3, channel_names=["ch0", "ch1"])

    class TestSlicingAccessors:

        def test_get_trial_returns_channels_by_time(self, dataset):
            result = dataset.get_trial(1)

            assert result.shape == (4, 100)
            np.testing.assert_array_equal(result, dataset.data[1])

        def test_get_trial_with_list_keeps_trial_axis(self, dataset):
            result = dataset.get_trial([0, 2])

            assert result.shape == (2, 4, 100)
            np.testing.assert_array_equal(result[0], dataset.data[0])
            np.testing.assert_array_equal(result[1], dataset.data[2])

class TestSlicingAccessors:

    def test_get_trial_returns_channels_by_time(self, dataset):
        result = dataset.get_trial(1)

        assert result.shape == (4, 100)
        np.testing.assert_array_equal(result, dataset.data[1])

    def test_get_trial_with_list_keeps_trial_axis(self, dataset):
        result = dataset.get_trial([0, 2])

        assert result.shape == (2, 4, 100)
        np.testing.assert_array_equal(result[0], dataset.data[0])
        np.testing.assert_array_equal(result[1], dataset.data[2])

    def test_get_trial_out_of_range_raises(self, dataset):
        with pytest.raises(IndexError):
            dataset.get_trial(99)

    def test_get_channel_returns_trials_by_time(self, dataset):
        result = dataset.get_channel(2)

        assert result.shape == (3, 100)
        np.testing.assert_array_equal(result, dataset.data[:, 2, :])

    def test_get_time_points_returns_trials_by_channels(self, dataset):
        result = dataset.get_time_points(50)

        assert result.shape == (3, 4)
        np.testing.assert_array_equal(result, dataset.data[:, :, 50])

    @pytest.mark.parametrize("accessor, arg", [
        ("get_trial", 0),
        ("get_trial", [0, 2]),
        ("get_channel", 2),
        ("get_time_points", 50),
    ])
    def test_result_does_not_share_memory_with_dataset(self, dataset,
                                                       accessor, arg):
        """
        np.shares_memory() is the direct test for view-vs-copy: it is True
        for a view into self.data and False for an independent array.
        """
        result = getattr(dataset, accessor)(arg)

        assert not np.shares_memory(result, dataset.data)

class TestZscoreData:

    def test_writes_to_data_zscored_not_data(self, dataset):

        original = dataset.data.copy()
        dataset.zscore_data(by_trial=True)

        np.testing.assert_array_equal(dataset.data, original)
        assert hasattr(dataset, "data_zscored")

    def test_attribute_is_written_after_the_call(self, dataset):
        """data_zscored is always initialized as None."""
        assert dataset.data_zscored is None

        dataset.zscore_data(by_trial=True)
        assert dataset.data_zscored is not None

    @pytest.mark.parametrize("by_trial", [True, False])
    def test_output_shape_matches_input(self, dataset, by_trial):
        dataset.zscore_data(by_trial=by_trial)

        assert dataset.data_zscored.shape == dataset.data.shape

    # by_trial=True: each channel normalised within its own trial

    def test_by_trial_gives_zero_mean_per_channel_per_trial(self, dataset):
        dataset.zscore_data(by_trial=True)

        # axis=1 is time, so every (trial, channel) row is standardised
        means = dataset.data_zscored.mean(axis=2)
        np.testing.assert_allclose(means, np.zeros((3, 4)), atol=1e-10)

    def test_by_trial_gives_unit_std_per_channel_per_trial(self, dataset):
        dataset.zscore_data(by_trial=True)

        stds = dataset.data_zscored.std(axis=2)
        np.testing.assert_allclose(stds, np.ones((3, 4)), atol=1e-10)

    def test_by_trial_matches_manual_per_trial_zscore(self, dataset):
        from scipy.stats import zscore

        dataset.zscore_data(by_trial=True)

        for i in range(dataset.n_trials):
            expected = zscore(dataset.data[i], axis=1)
            np.testing.assert_allclose(dataset.data_zscored[i], expected)

    # by_trial=False: each channel normalised across ALL trials

    def test_global_gives_zero_mean_per_channel_across_trials(self, dataset):
        """
        Pooling trials means each channel is standardised once over the whole
        recording, so per-trial means are NOT individually zero.
        """
        dataset.zscore_data(by_trial=False)

        # (trials, channels, time) → mean over trials AND time, per channel
        per_channel_mean = dataset.data_zscored.mean(axis=(0, 2))
        np.testing.assert_allclose(per_channel_mean, np.zeros(4), atol=1e-10)

    def test_global_gives_unit_std_per_channel_across_trials(self, dataset):
        dataset.zscore_data(by_trial=False)

        per_channel_std = dataset.data_zscored.std(axis=(0, 2))
        np.testing.assert_allclose(per_channel_std, np.ones(4), atol=1e-10)

    def test_global_reshape_preserves_trial_channel_alignment(self, dataset):
        """
        The method transposes to (channels, trials, time), flattens, then
        reshapes back.  A wrong axis order here would scramble the data, so
        verify by rebuilding the same transform independently.
        """
        from scipy.stats import zscore

        dataset.zscore_data(by_trial=False)

        temp = np.transpose(dataset.data, (1, 0, 2))
        flat = temp.reshape(temp.shape[0], -1)
        expected = np.transpose(zscore(flat, axis=1).reshape(temp.shape), (1, 0, 2))

        np.testing.assert_allclose(dataset.data_zscored, expected)

    def test_calling_twice_recomputes_from_raw_data(self, dataset):
        """
        Both branches read self.data, never self.data_zscored, so repeated
        calls are idempotent rather than cumulative.
        """
        dataset.zscore_data(by_trial=True)
        first = dataset.data_zscored.copy()

        dataset.zscore_data(by_trial=True)

        np.testing.assert_allclose(dataset.data_zscored, first)

class TestGetMeanAndStd:

    def test_returns_two_floats(self, dataset):
        mean_signal, std_signal = dataset._get_mean_and_std()

        assert isinstance(mean_signal, float)
        assert isinstance(std_signal, float)

    def test_values_are_positive(self, dataset):
        """Computed on absolute values, so both must be >= 0."""
        mean_signal, std_signal = dataset._get_mean_and_std()

        assert mean_signal >= 0
        assert std_signal >= 0

    def test_does_not_mutate_the_dataset(self, dataset):
        """A read-only helper: it must leave .data alone."""
        original = dataset.data.copy()

        dataset._get_mean_and_std()

        np.testing.assert_array_equal(dataset.data, original)

class TestFilterData:

    @patch(f"mne.filter.filter_data")
    def test_data_replaced_by_filter_output(self, mock_filter, dataset):
        expected = np.zeros_like(dataset.data)
        mock_filter.return_value = expected

        dataset._filter_data(s_freq=250.0, l_freq=1.0, h_freq=40.0)

        np.testing.assert_array_equal(dataset.data, expected)

    @patch(f"mne.filter.filter_data")
    def test_arguments_forwarded_to_mne(self, mock_filter, dataset):
        original = dataset.data
        mock_filter.return_value = np.zeros_like(original)

        dataset._filter_data(s_freq=250.0, l_freq=1.0, h_freq=40.0)

        _, kw = mock_filter.call_args
        np.testing.assert_array_equal(kw["data"], original)
        assert kw["sfreq"] == 250.0
        assert kw["l_freq"] == 1.0
        assert kw["h_freq"] == 40.0

    @patch(f"mne.filter.filter_data")
    def test_extra_kwargs_passed_through(self, mock_filter, dataset):
        mock_filter.return_value = np.zeros_like(dataset.data)

        dataset._filter_data(s_freq=250.0, l_freq=1.0, h_freq=40.0,
                            method="iir", phase="zero")

        _, kw = mock_filter.call_args
        assert kw["method"] == "iir"
        assert kw["phase"] == "zero"

    @patch(f"mne.filter.filter_data")
    def test_freq_is_written_correctly(self, mock_filter, dataset):

        mock_filter.return_value = np.zeros_like(dataset.data)

        dataset._filter_data(s_freq=250.0, l_freq=40.0, h_freq=1.0)

        assert dataset.fmin == 1.0
        assert dataset.fmax == 40.0

    @patch(f"mne.filter.filter_data")
    def test_none_cutoff_leaves_attribute_unchanged(self, mock_filter, dataset):
        """Only non-None cutoffs update the stored frequencies."""
        mock_filter.return_value = np.zeros_like(dataset.data)
        original_fmin = dataset.fmin

        dataset._filter_data(s_freq=250.0, l_freq=1.0, h_freq=None)

        assert dataset.fmin == original_fmin

class TestGetBinarizedData:

    @pytest.fixture
    def zscored_dataset(self):
        """A Dataset with zscore_data() already applied."""
        ds = make_dataset()
        ds.zscore_data(by_trial=True)
        return ds

    def test_requires_zscore_data_first(self, dataset):
        with pytest.raises(ValueError, match="The dataset needs to be z_scored"):
            dataset.get_binarized_data(threshold_method="fixed", threshold=2.0)

    def test_output_shape_matches_data(self, zscored_dataset):
        result = zscored_dataset.get_binarized_data(threshold_method="fixed",
                                                    threshold=2.0)

        assert result.shape == zscored_dataset.data.shape

    def test_output_contains_only_zeros_and_ones(self, zscored_dataset):
        result = zscored_dataset.get_binarized_data(threshold_method="fixed",
                                                    threshold=2.0)

        assert set(np.unique(result)).issubset({0, 1})

    def test_fixed_threshold_matches_manual_formula(self, zscored_dataset):
        result = zscored_dataset.get_binarized_data(threshold_method="fixed",
                                                    threshold=2.0)

        expected = np.where(np.abs(zscored_dataset.data_zscored) > 2.0, 1, 0)
        np.testing.assert_array_equal(result, expected)

    def test_fixed_uses_absolute_value(self, dataset_factory):
        """|-3| > 2 → 1, so strong negative deflections count as activations."""
        ds = dataset_factory(data=np.zeros((1, 2, 3)), n_channels=2)
        ds.data_zscored = np.array([[[-3.0, 1.0, 0.0],
                                     [3.0, 0.0, 0.0]]])

        result = ds.get_binarized_data(threshold_method="fixed", threshold=2.0)

        assert result[0, 0, 0] == 1
        assert result[0, 1, 0] == 1
        assert result[0, 0, 1] == 0

    def test_high_threshold_prints_warning(self, zscored_dataset, capsys):
        zscored_dataset.get_binarized_data(threshold_method="fixed",
                                           threshold=1000.0)

        assert "WARNING" in capsys.readouterr().out

    @pytest.mark.parametrize("bad_method", ["Fixed", "STD", "zscore", "", None])
    def test_invalid_method_raises_value_error(self, zscored_dataset, bad_method):
        with pytest.raises(ValueError, match="No valid thresholding method"):
            zscored_dataset.get_binarized_data(threshold_method=bad_method,
                                               threshold=2.0)

class TestCreateDatasetFromMneRaw:

    @pytest.fixture
    def mock_raw(self):
        """
        MagicMock(spec=mne.io.BaseRaw) makes isinstance() pass, so the Raw
        branch is taken without building a real Raw object.
        """
        mne = pytest.importorskip("mne")
        raw = MagicMock(spec=mne.io.BaseRaw)
        raw.get_data.return_value = np.random.default_rng(0).standard_normal((4, 500))
        raw.ch_names = ["ch0", "ch1", "ch2", "ch3"]
        raw.info = {"sfreq": 250.0, "lowpass": 40.0, "highpass": 0.5}
        return raw

    def test_returns_dataset(self, mock_raw):
        assert isinstance(create_dataset_from_mne(mock_raw), Dataset)

    def test_raw_gains_a_trial_axis(self, mock_raw):
        """(n_channels, n_times) → (1, n_channels, n_times)."""
        ds = create_dataset_from_mne(mock_raw)

        assert ds.data.shape == (1, 4, 500)
        assert ds.n_trials == 1

    def test_metadata_taken_from_info(self, mock_raw):
        ds = create_dataset_from_mne(mock_raw)

        assert ds.samp_freq == 250.0
        assert ds.fmin      == 0.5      # from info['highpass']
        assert ds.fmax      == 40.0     # from info['lowpass']

    def test_channel_names_copied(self, mock_raw):
        ds = create_dataset_from_mne(mock_raw)

        assert ds.channel_names == ["ch0", "ch1", "ch2", "ch3"]

    def test_raw_has_no_trial_labels(self, mock_raw):
        """Labels come from events, which a Raw object does not have."""
        ds = create_dataset_from_mne(mock_raw)

        assert ds.trial_labels is None

    def test_trial_ids_generated_by_dataset(self, mock_raw):
        ds = create_dataset_from_mne(mock_raw)

        assert ds.trial_ids == [0]

    def test_t_max_is_forwarded(self, mock_raw):

        ds = create_dataset_from_mne(mock_raw, t_max=100)

        assert ds.t_max == 100
        assert ds.data.shape[-1] == 100     # not truncated


class TestCreateDatasetFromMneEpochs:

    @pytest.fixture
    def mock_epochs(self):
        mne = pytest.importorskip("mne")
        epochs = MagicMock(spec=mne.BaseEpochs)
        epochs.get_data.return_value = \
            np.random.default_rng(1).standard_normal((3, 4, 100))
        epochs.ch_names = ["ch0", "ch1", "ch2", "ch3"]
        epochs.info = {"sfreq": 250.0, "lowpass": 40.0, "highpass": 0.5}
        epochs.events = np.array([[0,   0, 1],
                                  [100, 0, 2],
                                  [200, 0, 1]])
        epochs.__len__.return_value = 3
        return epochs

    def test_shape_preserved(self, mock_epochs):
        """Epochs data is already 3-D, so no axis is added."""
        ds = create_dataset_from_mne(mock_epochs)

        assert ds.data.shape == (3, 4, 100)
        assert ds.n_trials == 3

    def test_trial_labels_from_event_codes(self, mock_epochs):
        """Labels come from the last column of the events array."""
        ds = create_dataset_from_mne(mock_epochs)

        np.testing.assert_array_equal(ds.trial_labels, [1, 2, 1])

    def test_one_label_per_trial(self, mock_epochs):
        ds = create_dataset_from_mne(mock_epochs)

        assert len(ds.trial_labels) == ds.n_trials

    def test_trial_ids_match_trial_count(self, mock_epochs):
        ds = create_dataset_from_mne(mock_epochs)

        assert ds.trial_ids == [0, 1, 2]

    def test_metadata_taken_from_info(self, mock_epochs):
        ds = create_dataset_from_mne(mock_epochs)

        assert ds.samp_freq == 250.0
        assert ds.fmin      == 0.5
        assert ds.fmax      == 40.0

    @pytest.mark.parametrize("bad_input", [
        np.zeros((3, 4, 100)),
        {"data": None},
        [1, 2, 3],
        "raw_file.fif",
        42,
        None,
    ])
    def test_unsupported_type_raises_type_error(self, bad_input):
        with pytest.raises(TypeError, match="must be Raw or Epochs"):
            create_dataset_from_mne(bad_input)

class TestBuildDataset:

    @pytest.fixture
    def mock_epochs(self):
        mne = pytest.importorskip("mne")
        epochs = MagicMock(spec=mne.BaseEpochs)
        epochs.get_data.return_value = \
            np.random.default_rng(1).standard_normal((3, 4, 100))
        epochs.ch_names = ["ch0", "ch1", "ch2", "ch3"]
        epochs.info = {"sfreq": 250.0, "lowpass": 40.0, "highpass": 0.5}
        epochs.events = np.array([[0, 0, 1],
                                  [100, 0, 2],
                                  [200, 0, 1]])
        epochs.__len__.return_value = 3
        return epochs

    @pytest.fixture
    def mock_raw(self):
        """
        MagicMock(spec=mne.io.BaseRaw) makes isinstance() pass, so the Raw
        branch is taken without building a real Raw object.
        """
        mne = pytest.importorskip("mne")
        raw = MagicMock(spec=mne.io.BaseRaw)
        raw.get_data.return_value = np.random.default_rng(0).standard_normal((4, 500))
        raw.ch_names = ["ch0", "ch1", "ch2", "ch3"]
        raw.info = {"sfreq": 250.0, "lowpass": 40.0, "highpass": 0.5}
        return raw

    @pytest.fixture
    def valid_data_dict(self):

        np.random.seed(42)
        return {
            "data_array": np.random.randn(3, 3, 50),
            "samp_freq": 250.0,
            "fmin": 8.0,
            "fmax": 35.0,
            "channel_names": ["ch0", "ch1", "ch2"],
            "trial_labels": [0, 1, 0],
            "t_max": None
        }

    def test_correct_input_are_accepted(self, mock_epochs, mock_raw, valid_data_dict):
        d_epochs = _build_dataset(mock_epochs)
        d_raw = _build_dataset(mock_raw)
        d_dict = _build_dataset(valid_data_dict)
        assert isinstance(d_epochs, Dataset)
        assert isinstance(d_raw, Dataset)
        assert isinstance(d_dict, Dataset)

    @pytest.mark.parametrize("bad_input", [
        np.zeros((3, 4, 100)),
        [1, 2, 3],
        "raw_file.fif",
        42,
        None,
    ])
    def test_unsupported_type_raises_type_error(self, bad_input):
        with pytest.raises(TypeError, match="input_data must be an"):
            _build_dataset(bad_input)

    @pytest.mark.parametrize("key", [
        "data_array", "samp_freq", "fmin", "fmax",
        "channel_names", "trial_labels", "t_max"
    ])
    def test_missing_keys_raise_key_error(self, key):
        np.random.seed(42)
        d = {
            "data_array": np.random.randn(3, 3, 50),
            "samp_freq": 250.0,
            "fmin": 8.0,
            "fmax": 35.0,
            "channel_names": ["ch0", "ch1", "ch2"],
            "trial_labels": [0, 1, 0],
            "t_max": None
        }
        del d[key]
        with pytest.raises(KeyError, match="input dict is missing"):
            _build_dataset(d)