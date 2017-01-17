import theano
from blocks import initialization
from blocks.bricks import (Initializable, FeedforwardSequence, LinearMaxout,
                           Tanh, lazy, application, BatchNormalization, Linear,
                           NDimensionalSoftmax, Logistic, Softmax, Sequence, Rectifier)
from blocks.bricks.parallel import Fork
from blocks.utils import shared_floatx_nans
from blocks.roles import add_role, WEIGHT


class GatedBimodal(Initializable):

    u"""Gated Bimodal neural network.


    Parameters
    ----------
    dim : int
        The dimension of the hidden state.
    activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation. If ``None`` a
        :class:`.Tanh` brick is used.
    gate_activation : :class:`~.bricks.Brick` or None
        The brick to apply as activation for gates. If ``None`` a
        :class:`.Logistic` brick is used.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    @lazy(allocation=['dim'])
    def __init__(self, dim, activation=None, gate_activation=None,
                 **kwargs):
        self.dim = dim

        if not activation:
            activation = Tanh()
        if not gate_activation:
            gate_activation = Logistic()
        self.activation = activation
        self.gate_activation = gate_activation

        children = [activation, gate_activation]
        kwargs.setdefault('children', []).extend(children)
        super(GatedBimodal, self).__init__(**kwargs)

    def _allocate(self):
        self.W = shared_floatx_nans(
            (2 * self.dim, self.dim), name='input_to_gate')
        add_role(self.W, WEIGHT)
        self.parameters.append(self.W)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @application(inputs=['x_1', 'x_2'], outputs=['output', 'z'])
    def apply(self, x_1, x_2):
        x = theano.tensor.concatenate((x_1, x_2), axis=1)
        h = self.activation.apply(x)
        z = self.gate_activation.apply(x.dot(self.W))
        return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z


class GatedClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        (visual_init_range, textual_init_range, gbu_init_range,
         linear_range_1, linear_range_2, linear_range_3) = init_ranges
        visual_mlp = Sequence([
            BatchNormalization(input_dim=visual_dim).apply,
            Linear(visual_dim, hidden_size, use_bias=False,
                   weights_init=initialization.Uniform(width=visual_init_range)).apply,
        ], name='visual_mlp')
        textual_mlp = Sequence([
            BatchNormalization(input_dim=textual_dim).apply,
            Linear(textual_dim, hidden_size, use_bias=False,
                   weights_init=initialization.Uniform(width=textual_init_range)).apply,
        ], name='textual_mlp')

        gbu = GatedBimodal(hidden_size,
                           weights_init=initialization.Uniform(width=gbu_init_range))

        logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size, [
                                          linear_range_1, linear_range_2, linear_range_3])
        # logistic_mlp = Sequence([
        #    BatchNormalization(input_dim=hidden_size, name='bn1').apply,
        #    Linear(hidden_size, output_dim, name='linear_output', use_bias=False,
        #           weights_init=initialization.Uniform(width=linear_range_1)).apply,
        #    Logistic().apply
        #], name='logistic_mlp')

        children = [visual_mlp, textual_mlp, gbu, logistic_mlp]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', children)
        super(GatedClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat', 'z'])
    def apply(self, x_v, x_t):
        visual_mlp, textual_mlp, gbu, logistic_mlp = self.children
        visual_h = visual_mlp.apply(x_v)
        textual_h = textual_mlp.apply(x_t)
        h, z = gbu.apply(visual_h, textual_h)
        y_hat = logistic_mlp.apply(h)
        return y_hat, z


class MLPGenreClassifier(FeedforwardSequence, Initializable):

    def __init__(self, input_dim, output_dim, hidden_size, init_ranges, output_act=Logistic, **kwargs):
        linear1 = LinearMaxout(input_dim=input_dim, output_dim=hidden_size,
                               num_pieces=2, name='linear1')
        linear2 = LinearMaxout(input_dim=hidden_size, output_dim=hidden_size,
                               num_pieces=2, name='linear2')
        linear3 = Linear(input_dim=hidden_size, output_dim=output_dim)
        logistic = output_act()
        bricks = [
            BatchNormalization(input_dim=input_dim, name='bn1'),
            linear1,
            BatchNormalization(input_dim=hidden_size, name='bn2'),
            linear2,
            BatchNormalization(input_dim=hidden_size, name='bnl'),
            linear3,
            logistic]
        for init_range, b in zip(init_ranges, (linear1, linear2, linear3)):
            b.biases_init = initialization.Constant(0)
            b.weights_init = initialization.Uniform(width=init_range)

        kwargs.setdefault('use_bias', False)
        super(MLPGenreClassifier, self).__init__(
            [b.apply for b in bricks], **kwargs)


class LinearSumClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        (visual_range, textual_range, linear_range_1,
         linear_range_2, linear_range_3) = init_ranges
        visual_layer = FeedforwardSequence([
            BatchNormalization(input_dim=visual_dim).apply,
            LinearMaxout(input_dim=visual_dim, output_dim=hidden_size,
                         weights_init=initialization.Uniform(
                             width=visual_range),
                         use_bias=False,
                         biases_init=initialization.Constant(0),
                         num_pieces=2).apply],
            name='visual_layer')
        textual_layer = FeedforwardSequence([
            BatchNormalization(input_dim=textual_dim).apply,
            LinearMaxout(input_dim=textual_dim, output_dim=hidden_size,
                         weights_init=initialization.Uniform(
                             width=textual_range),
                         biases_init=initialization.Constant(0),
                         use_bias=False,
                         num_pieces=2).apply],
            name='textual_layer')
        logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size, [
                                          linear_range_1, linear_range_2, linear_range_3])
        # logistic_mlp = Sequence([
        #   BatchNormalization(input_dim=hidden_size, name='bn1').apply,
        #   Linear(hidden_size, output_dim, name='linear_output', use_bias=False,
        #          weights_init=initialization.Uniform(width=linear_range_1)).apply,
        #   Logistic().apply
        #], name='logistic_mlp')

        children = [visual_layer, textual_layer, logistic_mlp]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', children)
        super(LinearSumClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat'])
    def apply(self, x_v, x_t):
        visual_layer, textual_layer, logistic_mlp = self.children
        h = visual_layer.apply(x_v) + textual_layer.apply(x_t)
        return logistic_mlp.apply(h)


class ConcatenateClassifier(FeedforwardSequence, Initializable):

    def __init__(self, input_dim, output_dim, hidden_size, init_ranges, **kwargs):
        linear1 = LinearMaxout(input_dim=input_dim, output_dim=hidden_size,
                               num_pieces=2, name='linear1')
        linear2 = LinearMaxout(input_dim=hidden_size, output_dim=hidden_size,
                               num_pieces=2, name='linear2')
        linear3 = Linear(input_dim=hidden_size, output_dim=output_dim)
        logistic = Logistic()
        bricks = [
            linear1,
            BatchNormalization(input_dim=hidden_size, name='bn2'),
            linear2,
            BatchNormalization(input_dim=hidden_size, name='bnl'),
            linear3,
            logistic]
        for init_range, b in zip(init_ranges, (linear1, linear2, linear3)):
            b.biases_init = initialization.Constant(0)
            b.weights_init = initialization.Uniform(width=init_range)

        kwargs.setdefault('use_bias', False)
        super(ConcatenateClassifier, self).__init__(
            [b.apply for b in bricks], **kwargs)


class MoEClassifier(Initializable):

    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size, init_ranges, **kwargs):
        (visual_range, textual_range, linear_range_1,
         linear_range_2, linear_range_3) = init_ranges
        manager_dim = visual_dim + textual_dim
        visual_mlp = MLPGenreClassifier(visual_dim, output_dim, hidden_size, [
           linear_range_1, linear_range_2, linear_range_3], name='visual_mlp')
        textual_mlp = MLPGenreClassifier(textual_dim, output_dim, hidden_size, [
           linear_range_1, linear_range_2, linear_range_3], name='textual_mlp')
        # manager_mlp = MLPGenreClassifier(manager_dim, 2, hidden_size, [
        # linear_range_1, linear_range_2, linear_range_3], output_act=Softmax,
        # name='manager_mlp')
        bn = BatchNormalization(input_dim=manager_dim, name='bn3')
        manager_mlp = Sequence([
            Linear(manager_dim, 2, name='linear_output', use_bias=False,
                   weights_init=initialization.Uniform(width=linear_range_1)).apply,
        ], name='manager_mlp')
        fork = Fork(input_dim=manager_dim, output_dims=[2] * output_dim,
                    prototype=manager_mlp, output_names=['linear_' + str(i) for i in range(output_dim)])

        children = [visual_mlp, textual_mlp, fork, bn, NDimensionalSoftmax()]
        kwargs.setdefault('use_bias', False)
        kwargs.setdefault('children', children)
        super(MoEClassifier, self).__init__(**kwargs)

    @application(inputs=['x_v', 'x_t'], outputs=['y_hat'])
    def apply(self, x_v, x_t):
        visual_mlp, textual_mlp, fork, bn, softmax = self.children
        y_v, y_t = visual_mlp.apply(x_v), textual_mlp.apply(x_t)
        managers = fork.apply(bn.apply(theano.tensor.concatenate([x_v, x_t], axis=1)))
        g = softmax.apply(theano.tensor.stack(managers), extra_ndim=1)
        y = theano.tensor.stack([y_v, y_t])
        return (g.T * y).mean(axis=0) * 1.999 + 1e-5
