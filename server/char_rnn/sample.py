import tensorflow as tf
from read_utils import TextConverter
from model import CharRNN
import os
from IPython import embed

import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('lstm_size', 512, 'size of hidden state of lstm')
tf.flags.DEFINE_integer('num_layers', 1, 'number of lstm layers')
tf.flags.DEFINE_boolean('use_embedding', True, 'whether to use embedding')
tf.flags.DEFINE_integer('embedding_size', 256, 'size of embedding')
tf.flags.DEFINE_string('converter_path', 'model/jpfood/converter.pkl', 'model/name/converter.pkl')
tf.flags.DEFINE_string('checkpoint_path', 'model/jpfood', 'checkpoint path')
tf.flags.DEFINE_string('start_string', '', 'use this string to start generating')
tf.flags.DEFINE_integer('max_length', 30, 'max length to generate')

class Singleton(object):
  
  def __new__(cls, *args, **kwargs):
    if not hasattr(cls, '_instance'):
      orig = super(Singleton, cls)
      cls._instance = orig.__new__(cls, *args, **kwargs)
    return cls._instance


class Dianpin(Singleton):
    def __init__(self):
        self.text = ''
        self.tfmodel = None
        self.converter = None

    def model_built(self):#,vocab_size,sampling,lstm_size,num_layers,use_embedding,embedding_size):
        FLAGS.start_string = FLAGS.start_string.decode('utf-8')
        self.converter = TextConverter(filename=FLAGS.converter_path)
        if os.path.isdir(FLAGS.checkpoint_path):
            FLAGS.checkpoint_path =\
                tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        self.tfmodel = CharRNN(self.converter.vocab_size, sampling=True,
                    lstm_size=FLAGS.lstm_size, num_layers=FLAGS.num_layers,
                    use_embedding=FLAGS.use_embedding,
                    embedding_size=FLAGS.embedding_size)
        self.tfmodel.load(FLAGS.checkpoint_path)
        
    def final_predict(self):
        start = self.converter.text_to_arr(FLAGS.start_string)
        arr = self.tfmodel.sample(FLAGS.max_length, start, self.converter.vocab_size)
        return self.converter.arr_to_text(arr)

