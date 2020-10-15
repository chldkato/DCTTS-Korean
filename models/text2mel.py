from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import MAE
from models.modules import *
from util.hparams import *


class Text2Mel():

    def initialize(self, enc_input, dec_input, mel_target=None):
        is_training = True if mel_target is not None else False

        # TextEnc
        embedding = Embedding(symbol_length, embedding_dim)(enc_input)
        te = Conv1D(filters=2*d, kernel_size=1, padding='same', kernel_initializer='he_normal')(embedding)
        te = Activation('relu')(te)
        te = Conv1D(filters=2*d, kernel_size=1, padding='same', kernel_initializer='he_normal')(te)

        for _ in range(2):
            te = highway(te, filters=2*d, kernel_size=3, padding='same', dilation_rate=1)
            te = highway(te, filters=2*d, kernel_size=3, padding='same', dilation_rate=3)
            te = highway(te, filters=2*d, kernel_size=3, padding='same', dilation_rate=9)
            te = highway(te, filters=2*d, kernel_size=3, padding='same', dilation_rate=27)

        for _ in range(2):
            te = highway(te, filters=2*d, kernel_size=3, padding='same', dilation_rate=1)

        for _ in range(2):
            te = highway(te, filters=2*d, kernel_size=1, padding='same', dilation_rate=1)

        key, value = tf.split(te, num_or_size_splits=2, axis=-1)

        # AudioEnc
        ae = Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')(dec_input)
        ae = Activation('relu')(ae)
        ae = Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')(ae)
        ae = Activation('relu')(ae)
        ae = Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')(ae)

        for _ in range(2):
            ae = highway(ae, filters=d, kernel_size=3, padding='causal', dilation_rate=1)
            ae = highway(ae, filters=d, kernel_size=3, padding='causal', dilation_rate=3)
            ae = highway(ae, filters=d, kernel_size=3, padding='causal', dilation_rate=9)
            ae = highway(ae, filters=d, kernel_size=3, padding='causal', dilation_rate=27)

        for _ in range(2):
            ae = highway(ae, filters=d, kernel_size=3, padding='causal', dilation_rate=3)

        # Attention
        query = ae
        alignment = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) * tf.rsqrt(tf.cast(d, tf.float32)))
        context = tf.matmul(alignment, value)
        context = tf.concat([context, query], axis=-1)
        alignment = tf.transpose(alignment, [0, 2, 1])

        # AudioDec
        ad = Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')(context)
        ad = highway(ad, filters=d, kernel_size=3, padding='causal', dilation_rate=1)
        ad = highway(ad, filters=d, kernel_size=3, padding='causal', dilation_rate=3)
        ad = highway(ad, filters=d, kernel_size=3, padding='causal', dilation_rate=9)
        ad = highway(ad, filters=d, kernel_size=3, padding='causal', dilation_rate=27)

        for _ in range(2):
            ad = highway(ad, filters=d, kernel_size=3, padding='causal', dilation_rate=1)

        for _ in range(3):
            ad = Conv1D(filters=d, kernel_size=1, padding='causal', kernel_initializer='he_normal')(ad)
            ad = Activation('relu')(ad)

        ad = Conv1D(filters=mel_dim, kernel_size=1, padding='causal', kernel_initializer='he_normal')(ad)
        mel_output = Activation('sigmoid')(ad)

        self.enc_input = enc_input
        self.dec_input = dec_input
        self.mel_output = mel_output
        self.alignment = alignment
        self.mel_target = mel_target

        if is_training:
            self.loss = tf.reduce_mean(MAE(self.mel_target, self.mel_output))
            self.global_step = tf.Variable(0)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9)
            gv = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gv, global_step=self.global_step)
