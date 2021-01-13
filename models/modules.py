import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation
from copy import deepcopy
from util.hparams import *


def highway(input_data, filters, kernel_size, padding, dilation_rate):
    x = Conv1D(2*filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation_rate,
               kernel_initializer='he_normal')(input_data)
    h_1, h_2 = tf.split(x, num_or_size_splits=2, axis=-1)
    h_1 = Activation('sigmoid')(h_1)
    h_2 = Activation('relu')(h_2)
    return h_1 * h_2 + (1.0 - h_1) * input_data


def griffin_lim(spectrogram):
    spec = deepcopy(spectrogram)
    for i in range(50):
        est_wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
        est_stft = librosa.stft(est_wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        phase = est_stft / np.maximum(1e-8, np.abs(est_stft))
        spec = spectrogram * phase
    wav = librosa.istft(spec, hop_length=hop_length, win_length=win_length)
    return np.real(wav)
