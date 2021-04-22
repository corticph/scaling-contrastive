from functools import reduce

import tensorflow as tf

DEFAULT_FILTERS = [64, 128, 192, 256, 512, 512]
DEFAULT_KERNELS_SIZES = [10, 8, 4, 4, 4, 1]
DEFAULT_STRIDES = [5, 4, 2, 2, 2, 1]


class GroupNorm1D():

    def __init__(self, groups):
        """
        Group normalization paper:
        https://arxiv.org/abs/1803.08494
        
        Args:
            groups (int): Number of groups used to split the channel dimension. Must divide channels equally.
        """
        
        self.groups = groups
        self.built = False
        self.gamma = None
        self.beta = None

    def __call__(self, inputs, reuse=False):
        """
        Normalizes each group.
        
        Args:
            inputs (tf.Tensor): Inputs of shape BTC.
            reuse (bool): Whether to reuse variables defined within the variable scope.
        """

        with tf.variable_scope("group_norm", reuse=reuse):
            shape_op = tf.shape(inputs)
            N = shape_op[0] # dynamic
            T = shape_op[1] # dynamic
            C = inputs.shape[2]
            assert C % self.groups == 0
            inputs = tf.reshape(inputs, [N, T, self.groups, C // self.groups])
            mean, var = tf.nn.moments(inputs, [1, 3], keep_dims=True)
            inputs = (inputs - mean) / tf.sqrt(var + 1e-5)
            inputs = tf.reshape(inputs, [N, T, C])
            gamma = tf.get_variable("gamme", [1, 1, C], initializer=tf.ones_initializer())
            beta = tf.get_variable("beta", [1, 1, C], initializer=tf.zeros_initializer())
            outputs = inputs * gamma + beta
        return outputs


class EncoderConv1D():
    def __init__(self,
                 filters=DEFAULT_FILTERS,
                 kernel_sizes=DEFAULT_KERNELS_SIZES,
                 strides=DEFAULT_STRIDES,
                 groups=32):
        """
        Builds successive blocks of 1D convolutions of the form:
        --> conv1D (same padding)
        --> groupnorm (groups = 32)
        --> clipped ReLu (clip value = 5.0)
        --> dropout

        Input shape: batch x time x 1 (input depth defaults to 1)

        Args:
            filters (list): A list of ints defining number of filters for each layer.
            kernel_sizes (list): A list of 2-tuples defining kernel sizes for each layer.
            strides (list): A list of 2-tuples defining stride for each layer.
            dropout (float): The dropout rate.
        """
        assert len(filters) == len(kernel_sizes) == len(strides), \
        'Filters, kernel sizes and strides must be lists of the same length'
        
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.groups = groups
        
        self.num_layers = len(filters)
        self.temporal_reduction_factor = reduce(lambda x, y: x * y, strides)

        self.blocks = []
        for idx in range(self.num_layers):
            conv1d_layer = tf.layers.Conv1D(filters=self.filters[idx],
                                            kernel_size=self.kernel_sizes[idx],
                                            strides=self.strides[idx],
                                            padding='same',
                                            use_bias=False)
            gn_layer = GroupNorm1D(groups=self.groups)
            self.blocks.append((conv1d_layer, gn_layer))
            
    def __call__(self, inputs, seq_lens=None, reuse=False):
        """
        Computes the forward pass of the module.

        Args:
            inputs (tf.Tensor): Inputs of shape BFTC.
            seq_lens (tf.tensor): 1D tensor with sequence lengths for each batch example.
            reuse (bool): Whether to reuse variables defined within the block variable scopes.

        Returns:
            tf.Tensor: The outputs from the last convolutional block of shape BFTC.
            tf.Tensor: The adjusted sequence lengths will be returned.
        """

        for idx, (conv1d_layer, gn_layer) in enumerate(self.blocks):
            with tf.variable_scope(f"conv1d_block_{idx}", reuse=reuse):
                inputs = conv1d_layer(inputs)
                inputs = gn_layer(inputs)
                inputs = tf.keras.activations.relu(inputs, max_value=5.0)

        outputs = inputs

        return outputs, tf.cast(tf.ceil(seq_lens / self.temporal_reduction_factor), dtype=tf.int32)