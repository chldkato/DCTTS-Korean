from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.losses import MAE
from models.modules import *
from util.hparams import *


class SSRN():

    def initialize(self, mel_input, spec_target=None):
        is_training = True if spec_target is not None else False

        x = Conv1D(filters=c, kernel_size=1, padding='same', kernel_initializer='he_normal')(mel_input)

        x = highway(x, filters=c, kernel_size=3, padding='same', dilation_rate=1)
        x = highway(x, filters=c, kernel_size=3, padding='same', dilation_rate=3)

        for _ in range(2):
            x = tf.expand_dims(x, axis=1)
            x = Conv2DTranspose(filters=c, kernel_size=(1, 2), strides=(1, 2), padding='same', kernel_initializer='he_normal')(x)
            x = tf.squeeze(x, axis=1)
            x = highway(x, filters=c, kernel_size=3, padding='same', dilation_rate=1)
            x = highway(x, filters=c, kernel_size=3, padding='same', dilation_rate=3)

        x = Conv1D(filters=2*c, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        for _ in range(2):
            x = highway(x, filters=2*c, kernel_size=3, padding='same', dilation_rate=1)

        x = Conv1D(filters=f, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)

        for _ in range(2):
            x = Conv1D(filters=f, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
            x = Activation('relu')(x)

        x = Conv1D(filters=f, kernel_size=1, padding='same', kernel_initializer='he_normal')(x)
        spec_output = Activation('sigmoid')(x)

        self.mel_input = mel_input
        self.spec_output = spec_output
        self.spec_target = spec_target

        if is_training:
            self.loss = tf.reduce_mean(MAE(self.spec_target, self.spec_output))
            self.global_step = tf.Variable(0)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9)
            gv = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gv, global_step=self.global_step)
