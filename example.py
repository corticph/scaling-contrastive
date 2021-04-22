import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf

from model import EncoderConv1D, ContextLSTM, CPCModule
from model.utils import get_device, load_file

MODEL_PATH = "model/model_checkpoint/LSTM_BD_2x512"
TEST_FILE_PATH = "test.wav"

# build and initialize model (GPU support required)
encoder = EncoderConv1D()
context_fw = ContextLSTM()
context_bw = ContextLSTM()
cpc_module = CPCModule(encoder=encoder,
                       context_fw=context_fw,
                       context_bw=context_bw)

device_name = get_device()
with tf.device(device_name):
    with tf.variable_scope("Model"):
        x_ph = tf.placeholder(tf.float32, [None, None, 1]) # batch_size x time x depth
        x_seq_len_ph = tf.placeholder(tf.int32, [None])
        z_op, c_op, adj_seq_len_op = cpc_module(x_ph, seq_lens=x_seq_len_ph)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.per_process_gpu_memory_fraction = 1.0
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# load the LSTM-BD-2x512 model from the paper
var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Model') 
saver_op = tf.train.Saver(var_list=var_list)
saver_op.restore(sess, MODEL_PATH)

# test one a from librispeech dev-clean
x, x_seq_len = load_file(TEST_FILE_PATH)
feed = {x_ph: x, x_seq_len_ph: x_seq_len}
c = sess.run(c_op, feed_dict=feed)

print("\nSuccess!\n")