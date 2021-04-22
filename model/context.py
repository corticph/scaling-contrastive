import tensorflow as tf


class ContextLSTM():
    def __init__(self, num_units=512, num_layers=4):
        """
        Defines a simple block of LSTM layers.
        
        Args:
            num_units (int): Number of hidden state units for each LSTM layer.
            num_layers (int): Number of LSTM layers in the block.
        """

        self.num_units = num_units
        self.num_layers = num_layers
        
        self.layers = []
        for idx in range(self.num_layers):
            lstm_layer = tf.contrib.cudnn_rnn.CudnnLSTM(num_units=self.num_units,
                                                        direction='unidirectional',
                                                        num_layers=1)
            self.layers.append(lstm_layer)

    def __call__(self, inputs, seq_lens, reuse=False):
        """
        Args:
            inputs (tf.Tensor): Inputs of shape time x batch_size x features.
            seq_lens (tf.Tensor): Only required for the sake of generality.

        Returns:
            tf.Tensor: The outputs from the last LSTM layer of shape time x batch_size x features.
            tf.Tensor: The output sequence lengths. Not altered for this module.
        """
        for idx, lstm_layer in enumerate(self.layers):
            with tf.variable_scope(f'rnn_layer_{idx}', reuse=reuse):
                inputs, _ = lstm_layer(inputs)
        outputs = inputs
        return outputs, seq_lens
