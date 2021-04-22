from functools import reduce

import tensorflow as tf

class CPCModule():
    def __init__(self, encoder, context_fw, context_bw):
        """
        A CPC module that combines encoder and two context networks.
        
        Args:
            encoder: The encoder module.
            context_fw: The forward context network.
            context_bw: The backward context network.
        """

        self.encoder = encoder
        self.context_fw = context_fw
        self.context_bw = context_bw

    def __call__(self, inputs, seq_lens, reuse=False):
        """
        Extracts latent (z) and context (c) representations.
        
        Args:
            inputs (tf.Tensor): Inputs with shape according to the encoder.
            seq_lens (tf.Tensor): 1D tensor with sequence lengths for each batch example.
        
        Returns:
            tf.Tensor: The latent representations of shape time x batch_size x encoder_features.
            tf.Tensor: The latent representations of shape time x batch_size x context_features.
            tf.Tensor: The adjusted sequence lengths will be returned.
        """
        
        with tf.variable_scope('cpc_encoder', reuse=reuse):
            z, adj_seq_lens = self.encoder(inputs=inputs, seq_lens=seq_lens, reuse=reuse)
            z = tf.transpose(z, [1, 0, 2]) # set time major
        
        with tf.variable_scope('cpc_context', reuse=reuse):

            with tf.variable_scope('forward', reuse=reuse):
                c_fw, _ = self.context_fw(z, adj_seq_lens, reuse=reuse)
        
            with tf.variable_scope('backward', reuse=reuse):
                z_rev = tf.reverse_sequence(input=z, seq_lengths=adj_seq_lens, seq_axis=0, batch_axis=1)
                c_bw_rev, _ = self.context_bw(z_rev, adj_seq_lens, reuse=reuse)
                c_bw = tf.reverse_sequence(input=c_bw_rev, seq_lengths=adj_seq_lens, seq_axis=0, batch_axis=1)
        
        c = tf.concat([c_fw, c_bw], axis=-1)
        
        return z, c, adj_seq_lens