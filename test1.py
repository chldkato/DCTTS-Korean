import numpy as np
import tensorflow as tf
import os, argparse
from jamo import hangul_to_jamo
from models.text2mel import Text2Mel
from util.text import text_to_sequence, sequence_to_text
from util.plot_alignment import plot_alignment
from util.hparams import *


sentences = [
  '정말로 사랑한담 기다려주세요'
]

checkpoint_dir = './checkpoint'


class Synthesizer:
    def load(self, step):
        enc_input = tf.placeholder(tf.int32, [1, None])
        dec_input = tf.placeholder(tf.float32, [1, None, mel_dim])

        self.model = Text2Mel()
        self.model.initialize(enc_input, dec_input)

        self.enc_input = self.model.enc_input[0]
        self.mel_output = self.model.mel_output[0]
        self.alignment = self.model.alignment[0]

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, os.path.join(checkpoint_dir, '1/model.ckpt-{}'.format(step)))


    def synthesize(self, args, text, idx):
        seq = text_to_sequence(text)
        dec_input = np.zeros((1, max_T, mel_dim), dtype='float32')
        pred = []
        for i in range(1, max_T+1):
            mel_out, alignment = self.session.run([self.mel_output, self.alignment],
                                                  feed_dict={self.model.enc_input: [np.asarray(seq, dtype=np.int32)],
                                                             self.model.dec_input: dec_input})
            if i < max_T:
                dec_input[:, i, :] = mel_out[i - 1, :]
            pred.extend(mel_out[i-1 : i, :])

        np.save(os.path.join(args.save_dir, 'mel-{}'.format(idx)), pred, allow_pickle=False)

        input_seq = sequence_to_text(seq)
        alignment_dir = os.path.join(args.save_dir, 'align-{}.png'.format(idx))
        plot_alignment(alignment, alignment_dir, input_seq)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', required=True)
    parser.add_argument('--save_dir', default='./output')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    synth = Synthesizer()
    synth.load(args.step)
    for i, text in enumerate(sentences):
        jamo = ''.join(list(hangul_to_jamo(text)))
        synth.synthesize(args, jamo, i)
