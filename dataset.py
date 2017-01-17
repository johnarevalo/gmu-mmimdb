import h5py
import json
from fuel.datasets import H5PYDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme, BatchScheme
from fuel.transformers import Padding
from fuel.transformers.defaults import uint8_pixels_to_floatX
from fuel.transformers import AgnosticSourcewiseTransformer
from fuel.utils import find_in_data_path
from sklearn import metrics


class SequenceTransposer(AgnosticSourcewiseTransformer):

    def __init__(self, data_stream, **kwargs):
        if data_stream.axis_labels:
            kwargs.setdefault('axis_labels', data_stream.axis_labels.copy())
        super(SequenceTransposer, self).__init__(
            data_stream, data_stream.produces_examples, **kwargs)

    def transform_any_source(self, source_data, _):
        if source_data.ndim == 2:
            return source_data.T
        elif source_data.ndim == 3:
            return source_data.transpose(1, 0, 2)
        else:
            raise ValueError('Invalid dimensions of this source.')


class MMImdbDataset(H5PYDataset):

    filename = 'multimodal_imdb.hdf5'
    default_transformers = (
        (Padding, [], {'mask_sources': ('sequences',)}),
        (SequenceTransposer, [], {
            'which_sources': ('sequences', 'sequences_mask')})
    ) + uint8_pixels_to_floatX(('images',))

    def __init__(self, which_sets, **kwargs):

        #kwargs.setdefault('file_or_path', MMImdbDataset.get_filepath())
        kwargs.setdefault('sources', ('features', 'genres'))
        super(MMImdbDataset, self).__init__(
            which_sets=which_sets,
            **kwargs)

    @staticmethod
    def get_filepath(filename=None):
        if filename is None:
            filename = MMImdbDataset.filename
        return find_in_data_path(filename)

    def create_stream(self, batch_size=None):
        if batch_size is None:
            batch_size = self.num_examples
        return DataStream.default_stream(dataset=self, iteration_scheme=ShuffledScheme(
            examples=self.num_examples, batch_size=batch_size))

    def get_target_names(filename=None):
        if filename is None:
            filename = MMImdbDataset.get_filepath()
        with h5py.File(filename, 'r') as f:
            target_names = json.loads(f['genres'].attrs['target_names'])
        return target_names


def report_performance(y_true, y_prob, threshold, print_results=True, multilabel=True):
    y_pred = y_prob > threshold
    results = {}
    averages = ('micro', 'macro', 'weighted', 'samples')
    if multilabel:
        acc = metrics.accuracy_score(y_true, y_pred)
    else:
        acc = metrics.accuracy_score(
            y_true.argmax(axis=1), y_prob.argmax(axis=1))
    for average in averages:
        results[average] = metrics.precision_recall_fscore_support(y_true, y_pred, average=average)[:3] + (
            metrics.roc_auc_score(y_true, y_prob, average=average),
            metrics.hamming_loss(y_true, y_pred),
            acc)

    if print_results:
        print('average\tprecisi\trecall\tf_score\tauc\thamming\taccuracy')
        for avg, vals in results.items():
            print('{0:.7}\t{1:0.3f}\t{2:0.3f}\t{3:0.3f}\t{4:0.3f}\t{5:0.3f}\t{6:0.3f}'.format(
                avg, *vals))
        target_names = MMImdbDataset.get_target_names('multimodal_imdb.hdf5')
        print(metrics.classification_report(
            y_true, y_pred, target_names=target_names))
    return results
