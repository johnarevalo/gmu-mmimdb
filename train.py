import os
from abc import ABCMeta, abstractmethod
from model import *
from theano import tensor
from dataset import MMImdbDataset, report_performance
from blocks.filter import VariableFilter
from blocks.serialization import load_parameters
from blocks.monitoring.evaluators import DatasetEvaluator
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.graph import ComputationGraph, batch_normalization
from blocks.extensions.stopping import EarlyStopping
from experiment import Experiment


class Trainer(object):

    def __init__(self, params):
        self.params = params
        self.train_stream = MMImdbDataset(('train',),
                                          file_or_path='multimodal_imdb.hdf5',
                                          load_in_memory=self.params[
                                              'load_in_memory'],
                                          sources=tuple(
                                              self.params['sources']),
                                          ).create_stream(self.params['batch_size'])
        self.dev_stream = MMImdbDataset(('dev',), load_in_memory=self.params['load_in_memory'],
                                          file_or_path='multimodal_imdb.hdf5',
                                        sources=tuple(self.params['sources']),
                                        ).create_stream()
        self.test_stream = MMImdbDataset(('test',), load_in_memory=self.params['load_in_memory'],
                                          file_or_path='multimodal_imdb.hdf5',
                                         sources=tuple(self.params['sources']),
                                         ).create_stream()

    def train(self):
        error = tensor.neq(self.y.flatten(), self.y_hat.flatten() > 0.5).mean()
        error.name = 'error'
        self.error = error
        experiment = Experiment(self.params['model_name'], self.train_stream)
        experiment.cost = self.cost
        experiment.set_adam(self.params['learning_rate'])
        experiment.add_printing(after_epoch=True)
        experiment.monitor_f_score(self.y, self.y_hat, average='macro',
                                   threshold=self.params['threshold'])
        experiment.monitor_auc_score(self.y, self.y_hat, average='macro')
        experiment.add_timing()
        experiment.extensions.append(EarlyStopping('dev_f_score', epochs=self.params['n_epochs'],
                                                   choose_best=max))
        weights = VariableFilter(theano_name='W')(experiment.cg.variables)
        experiment.regularize_max_norm(self.params['max_norms'], weights)
        experiment.apply_dropout(self.params['dropout'])
        experiment.track_best(
            'dev_f_score', save_path=self.params['model_name'] + '.tar', choose_best=max)
        experiment.track_best('dev_cost', save_path=self.params[
                              'model_name'] + '_cost.tar')
        experiment.plot_channels(channels=[['tra_f_score', 'dev_f_score'],
                                           ['tra_cost', 'dev_cost'],
                                           ],
                                 url_bokeh='http://localhost:5006/',
                                 before_first_epoch=True, after_epoch=True)
        experiment.add_monitored_vars([error])
        experiment.add_norm_grads_vars()
        experiment.monitor_stream(
            self.train_stream, prefix='tra', after_epoch=True)
        experiment.monitor_stream(self.dev_stream, prefix='dev')
        self.experiment = experiment

        print('# of params for the model: {0}'.format(
            experiment.get_num_params()))
        main_loop = experiment.get_main_loop()
        if not os.path.isfile(self.params['model_name'] + '.tar'):
            main_loop.run()

        with open(self.params['model_name'] + '.tar', "rb") as f:
            print('loading saved model...')
            main_loop.model.set_parameter_values(load_parameters(f))

    def evaluate(self):
        evaluator = DatasetEvaluator(
            [self.cost, self.error] + self.experiment.get_quantitites_vars())
        print('subset\tcost\terror\tf_score\tauc')
        for split in ['train', 'dev', 'test']:
            stream = getattr(self, split + '_stream')
            print('{0}\t{cost}\t{error}\t{f_score}\t{auc}'.format(
                split, **evaluator.evaluate(stream)))

        y_prob, y_test = self.get_targets(self.test_stream)
        report_performance(y_test, y_prob, self.params['threshold'])


class MaxoutMLP(Trainer):

    def __init__(self, params, feature_source, input_dim):
        super(MaxoutMLP, self).__init__(params)
        self.x = tensor.matrix(feature_source, dtype='float32')
        self.y = tensor.matrix('genres', dtype='int32')
        mlp = MLPGenreClassifier(input_dim,
                                 self.params['n_classes'],
                                 self.params['hidden_size'],
                                 self.params['init_ranges'])
        mlp.initialize()
        with batch_normalization(mlp):
            self.y_hat = mlp.apply(self.x)
        self.cost = BinaryCrossEntropy().apply(self.y, self.y_hat)

    def get_targets(self, stream):
        fn = ComputationGraph(self.y_hat).get_theano_function()
        X_test, y_test = next(stream.get_epoch_iterator())
        y_prob = fn(X_test)[0]
        return y_prob, y_test


class MaxoutMLP_w2v(MaxoutMLP):

    def __init__(self, params):
        super(MaxoutMLP_w2v, self).__init__(params,
                                            feature_source='features',
                                            input_dim=params['textual_dim'])


class MaxoutMLP_VGG(MaxoutMLP):

    def __init__(self, params):
        super(MaxoutMLP_VGG, self).__init__(params,
                                            feature_source='vgg_features',
                                            input_dim=params['visual_dim'])


class GatedTrainer(Trainer):

    def __init__(self, params):
        super(GatedTrainer, self).__init__(params)
        self.x_v = tensor.matrix('vgg_features', dtype='float32')
        self.x_t = tensor.matrix('features', dtype='float32')
        self.y = tensor.matrix('genres', dtype='int32')
        model = GatedClassifier(params['visual_dim'],
                                params['textual_dim'],
                                params['n_classes'],
                                params['hidden_size'],
                                params['init_ranges'])
        model.initialize()
        with batch_normalization(model):
            self.y_hat, self.z = model.apply(self.x_v, self.x_t)
        self.cost = BinaryCrossEntropy().apply(self.y, self.y_hat)

    def get_targets(self, stream):
        fn = ComputationGraph(self.y_hat).get_theano_function()
        y_test, X_v, X_t = next(stream.get_epoch_iterator())
        y_prob = fn(X_t, X_v)[0]
        return y_prob, y_test


class LinearSumTrainer(Trainer):

    def __init__(self, params):
        super(LinearSumTrainer, self).__init__(params)
        self.x_v = tensor.matrix('vgg_features', dtype='float32')
        self.x_t = tensor.matrix('features', dtype='float32')
        self.y = tensor.matrix('genres', dtype='int32')
        model = LinearSumClassifier(params['visual_dim'],
                                    params['textual_dim'],
                                    params['n_classes'],
                                    params['hidden_size'],
                                    params['init_ranges'])
        model.initialize()
        with batch_normalization(model):
            self.y_hat = model.apply(self.x_v, self.x_t)
        self.cost = BinaryCrossEntropy().apply(self.y, self.y_hat)

    def get_targets(self, stream):
        fn = ComputationGraph(self.y_hat).get_theano_function()
        y_test, X_v, X_t = next(stream.get_epoch_iterator())
        y_prob = fn(X_t, X_v)[0]
        return y_prob, y_test


class ConcatenateTrainer(Trainer):

    def __init__(self, params):
        super(ConcatenateTrainer, self).__init__(params)
        x_v = tensor.matrix('vgg_features', dtype='float32')
        x_t = tensor.matrix('features', dtype='float32')
        self.x = tensor.concatenate([x_v, x_t], axis=1)
        self.y = tensor.matrix('genres', dtype='int32')
        input_dim = params['visual_dim'] + params['textual_dim']
        mlp = MLPGenreClassifier(input_dim,
                                 self.params['n_classes'],
                                 self.params['hidden_size'],
                                 self.params['init_ranges'])
        mlp.initialize()
        with batch_normalization(mlp):
            self.y_hat = mlp.apply(self.x)
        self.cost = BinaryCrossEntropy().apply(self.y, self.y_hat)

    def get_targets(self, stream):
        fn = ComputationGraph(self.y_hat).get_theano_function()
        y_test, X_v, X_t = next(stream.get_epoch_iterator())
        y_prob = fn(X_v, X_t)[0]
        return y_prob, y_test


class MoETrainer(Trainer):

    def __init__(self, params):
        super(MoETrainer, self).__init__(params)
        self.x_v = tensor.matrix('vgg_features', dtype='float32')
        self.x_t = tensor.matrix('features', dtype='float32')
        self.y = tensor.matrix('genres', dtype='int32')
        model = MoEClassifier(params['visual_dim'],
                              params['textual_dim'],
                              params['n_classes'],
                              params['hidden_size'],
                              params['init_ranges'])
        model.initialize()
        with batch_normalization(model):
            self.y_hat = model.apply(self.x_v, self.x_t)
        self.cost = BinaryCrossEntropy().apply(self.y, self.y_hat)

    def get_targets(self, stream):
        fn = ComputationGraph(self.y_hat).get_theano_function()
        y_test, X_v, X_t = next(stream.get_epoch_iterator())
        y_prob = fn(X_t, X_v)[0]
        return y_prob, y_test
