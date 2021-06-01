from numpy.core.numeric import Inf
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.keras.backend import dtype
import tensorflow_addons as tfa
import numpy as np
from scipy import stats
import math


def get_mel_spectrogram(waveforms, invert_mu_law=True, mu=65535., jitter=False, max_jitter_steps=60):
    """Computes mel-spectrograms for the given waveforms.
    Args:
    waveforms: a tf.Tensor corresponding to a batch of waveforms
    sampled at 24 kHz.
    (dtype=tf.float32, shape=[N, sequence_length])
    invert_mu_law: whether to apply mu-law inversion to the input waveforms.
    In EATS both the real data and generator outputs are mu-law'ed, so this is
    always set to True.
    mu: The mu value used if invert_mu_law=True (ignored otherwise).
    jitter: whether to apply random jitter to the input waveforms before
    computing spectrograms. Set to True only for GT spectrograms input to the
    prediction loss.
    max_jitter_steps: maximum number of steps by which the input waveforms are
    randomly jittered if jitter=True (ignored otherwise).
    Returns:
    A 3D tensor with spectrograms for the corresponding input waveforms.
    (dtype=tf.float32,
    shape=[N, num_frames=ceil(sequence_length/1024), num_bins=80])
    """
    waveforms.shape.assert_has_rank(2)
    t = waveforms
    if jitter:
        assert max_jitter_steps >= 0
        crop_shape = [t.shape[1]]
        t = tf.pad(t, [[0, 0], [max_jitter_steps, max_jitter_steps]])
        # Jitter independently for each batch item.
        t = tf.map_fn(lambda ti: tf.image.random_crop(ti, crop_shape), t)
    if invert_mu_law:
        t = tf.sign(t) / mu * ((1 + mu)**tf.abs(t) - 1)
    t = tf.signal.stft(t, frame_length=2048, frame_step=1024, pad_end=True)
    t = tf.abs(t)
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=80, num_spectrogram_bins=t.shape[-1],
    sample_rate=24000., lower_edge_hertz=80., upper_edge_hertz=7600.)
    t = tf.tensordot(t, mel_weight_matrix, axes=1)
    t = tf.math.log(1. + 10000.*t)
    return t

def soft_minimum(values, temperature):
    """Compute the soft minimum with the given temperature."""
    return -temperature * math.log(sum(math.exp(-values / temperature)))

def skew_matrix(x):
    """Skew a matrix so that the diagonals become the rows."""
    height, width = x.shape
    y = np.zeros(height + width - 1, width)
    for i in range(height + width - 1):
        for j in range(width): # Shift each column j down by j steps.
            y[i, j] = x[np.clip(i - j, 0, height - 1), j]
    return y
def spectrogram_dtw_error(spec_a, spec_b, warp_penalty=1.0, temperature=0.01):
    """Compute DTW error given a pair of spectrograms."""
    # Compute cost matrix.
    diffs = abs(spec_a[None, :, :] - spec_b[:, None, :])
    costs = np.mean(diffs, axis=-1) # pairwise L1 cost, square the diffs for L2.
    size = costs.shape[-1]
    # Initialise path costs.
    path_cost = Inf * np.ones(size + 1)
    path_cost_prev = Inf * np.ones(size + 1)
    path_cost_prev[0] = 0.0
    # Aggregate path costs from cost[0, 0] to cost[-1, -1].
    cost = skew_matrix(costs) # Shape is now (2 * size - 1, size).
    for i in range(2 * size - 1):
        directions = [path_cost_prev[:-1],
        path_cost[1:] + warp_penalty,
        path_cost[:-1] + warp_penalty]
        path_cost_next = cost[i] + soft_minimum(directions, temperature)
        # Replace soft minimum with regular minimum for regular DTW.
        path_cost_next = np.concatenate([[Inf], path_cost_next])
        path_cost, path_cost_prev = path_cost_next, path_cost
    return path_cost[-1]


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = tf.cast(mask, keras.backend.floatx())
            inputs *= tf.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)

class EATSAligner(keras.Model):
    def __init__(self, token_vocab_size):
        super(EATSAligner, self).__init__()
        self.embd = keras.layers.Embedding(token_vocab_size, 256, mask_zero=True)

        self.bn1 = ConditionalBatchNorm(256)
        self.rel1 = keras.layers.Activation('relu')
        self.conv1 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn2 = ConditionalBatchNorm(256)
        self.rel2 = keras.layers.Activation('relu')
        self.conv2 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn3 = ConditionalBatchNorm(256)
        self.rel3 = keras.layers.Activation('relu')
        self.conv3 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn4 = ConditionalBatchNorm(256)
        self.rel4 = keras.layers.Activation('relu')
        self.conv4 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn5 = ConditionalBatchNorm(256)
        self.rel5 = keras.layers.Activation('relu')
        self.conv5 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn6 = ConditionalBatchNorm(256)
        self.rel6 = keras.layers.Activation('relu')
        self.conv6 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())
        
        self.bn7 = ConditionalBatchNorm(256)
        self.rel7 = keras.layers.Activation('relu')
        self.conv7 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn8 = ConditionalBatchNorm(256)
        self.rel8 = keras.layers.Activation('relu')
        self.conv8 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn9 = ConditionalBatchNorm(256)
        self.rel9 = keras.layers.Activation('relu')
        self.conv9 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn10 = ConditionalBatchNorm(256)
        self.rel10 = keras.layers.Activation('relu')
        self.conv10 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn11 = ConditionalBatchNorm(256)
        self.rel11 = keras.layers.Activation('relu')
        self.conv11 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn12 = ConditionalBatchNorm(256)
        self.rel12 = keras.layers.Activation('relu')
        self.conv12 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn13 = ConditionalBatchNorm(256)
        self.rel13 = keras.layers.Activation('relu')
        self.conv13 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn14 = ConditionalBatchNorm(256)
        self.rel14 = keras.layers.Activation('relu')
        self.conv14 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn15 = ConditionalBatchNorm(256)
        self.rel15 = keras.layers.Activation('relu')
        self.conv15 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn16 = ConditionalBatchNorm(256)
        self.rel16 = keras.layers.Activation('relu')
        self.conv16 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn17 = ConditionalBatchNorm(256)
        self.rel17 = keras.layers.Activation('relu')
        self.conv17 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn18 = ConditionalBatchNorm(256)
        self.rel18 = keras.layers.Activation('relu')
        self.conv18 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn19 = ConditionalBatchNorm(256)
        self.rel19 = keras.layers.Activation('relu')
        self.conv19 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn20 = ConditionalBatchNorm(256)
        self.rel20 = keras.layers.Activation('relu')
        self.conv20 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn21 = ConditionalBatchNorm(256)
        self.rel21 = keras.layers.Activation('relu')
        self.conv21 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn22 = ConditionalBatchNorm(256)
        self.rel22 = keras.layers.Activation('relu')
        self.conv22 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn23 = ConditionalBatchNorm(256)
        self.rel23 = keras.layers.Activation('relu')
        self.conv23 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn24 = ConditionalBatchNorm(256)
        self.rel24 = keras.layers.Activation('relu')
        self.conv24 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn25 = ConditionalBatchNorm(256)
        self.rel25 = keras.layers.Activation('relu')
        self.conv25 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn26 = ConditionalBatchNorm(256)
        self.rel26 = keras.layers.Activation('relu')
        self.conv26 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn27 = ConditionalBatchNorm(256)
        self.rel27 = keras.layers.Activation('relu')
        self.conv27 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn28 = ConditionalBatchNorm(256)
        self.rel28 = keras.layers.Activation('relu')
        self.conv28 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn29 = ConditionalBatchNorm(256)
        self.rel29 = keras.layers.Activation('relu')
        self.conv29 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn30 = ConditionalBatchNorm(256)
        self.rel30 = keras.layers.Activation('relu')
        self.conv30 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn31 = ConditionalBatchNorm(256)
        self.rel31 = keras.layers.Activation('relu')
        self.conv31 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn32 = ConditionalBatchNorm(256)
        self.rel32 = keras.layers.Activation('relu')
        self.conv32 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn33 = ConditionalBatchNorm(256)
        self.rel33 = keras.layers.Activation('relu')
        self.conv33 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn34 = ConditionalBatchNorm(256)
        self.rel34 = keras.layers.Activation('relu')
        self.conv34 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn35 = ConditionalBatchNorm(256)
        self.rel35 = keras.layers.Activation('relu')
        self.conv35 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn36 = ConditionalBatchNorm(256)
        self.rel36 = keras.layers.Activation('relu')
        self.conv36 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn37 = ConditionalBatchNorm(256)
        self.rel37 = keras.layers.Activation('relu')
        self.conv37 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn38 = ConditionalBatchNorm(256)
        self.rel38 = keras.layers.Activation('relu')
        self.conv38 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn39 = ConditionalBatchNorm(256)
        self.rel39 = keras.layers.Activation('relu')
        self.conv39 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn40 = ConditionalBatchNorm(256)
        self.rel40 = keras.layers.Activation('relu')
        self.conv40 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn41 = ConditionalBatchNorm(256)
        self.rel41 = keras.layers.Activation('relu')
        self.conv41 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn42 = ConditionalBatchNorm(256)
        self.rel42 = keras.layers.Activation('relu')
        self.conv42 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn43 = ConditionalBatchNorm(256)
        self.rel43 = keras.layers.Activation('relu')
        self.conv43 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn44 = ConditionalBatchNorm(256)
        self.rel44 = keras.layers.Activation('relu')
        self.conv44 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn45 = ConditionalBatchNorm(256)
        self.rel45 = keras.layers.Activation('relu')
        self.conv45 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn46 = ConditionalBatchNorm(256)
        self.rel46 = keras.layers.Activation('relu')
        self.conv46 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn47 = ConditionalBatchNorm(256)
        self.rel47 = keras.layers.Activation('relu')
        self.conv47 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn48 = ConditionalBatchNorm(256)
        self.rel48 = keras.layers.Activation('relu')
        self.conv48 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn49 = ConditionalBatchNorm(256)
        self.rel49 = keras.layers.Activation('relu')
        self.conv49 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn50 = ConditionalBatchNorm(256)
        self.rel50 = keras.layers.Activation('relu')
        self.conv50 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn51 = ConditionalBatchNorm(256)
        self.rel51 = keras.layers.Activation('relu')
        self.conv51 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn52 = ConditionalBatchNorm(256)
        self.rel52 = keras.layers.Activation('relu')
        self.conv52 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn53 = ConditionalBatchNorm(256)
        self.rel53 = keras.layers.Activation('relu')
        self.conv53 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn54 = ConditionalBatchNorm(256)
        self.rel54 = keras.layers.Activation('relu')
        self.conv54 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn55 = ConditionalBatchNorm(256)
        self.rel55 = keras.layers.Activation('relu')
        self.conv55 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=1, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn56 = ConditionalBatchNorm(256)
        self.rel56 = keras.layers.Activation('relu')
        self.conv56 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn57 = ConditionalBatchNorm(256)
        self.rel57 = keras.layers.Activation('relu')
        self.conv57 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn58 = ConditionalBatchNorm(256)
        self.rel58 = keras.layers.Activation('relu')
        self.conv58 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn59 = ConditionalBatchNorm(256)
        self.rel59 = keras.layers.Activation('relu')
        self.conv59 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=16, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn60 = ConditionalBatchNorm(256)
        self.rel60 = keras.layers.Activation('relu')
        self.conv60 = MaskedConv1D(filters=256, kernel_size=3, padding='same', dilation_rate=32, kernel_initializer=tf.keras.initializers.orthogonal())

        self.bn61 = ConditionalBatchNorm(256)
        self.rel61 = keras.layers.Activation('relu')
        self.conv61 = keras.layers.Conv1D(256, 1, padding='same', kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn62 = ConditionalBatchNorm(256)
        self.rel62 = keras.layers.Activation('relu')
        self.conv62 = keras.layers.Conv1D(1, 1, padding='same', kernel_initializer=tf.keras.initializers.orthogonal())

        self.rel63 = keras.layers.Activation('relu')

    def call(self, token_sequences, noise, training=False):
        embedded_tokens = self.embd(token_sequences)
        mask = self.embd.compute_mask(token_sequences)
        x = embedded_tokens
        x = self.bn1(x, noise, training)
        x = self.rel1(x)
        x = self.conv1(x, mask)
        x = self.bn2(x, noise, training)
        x = self.rel2(x)
        x = self.conv2(x, mask)
        x = self.bn3(x, noise, training)
        x = self.rel3(x)
        x = self.conv3(x, mask)
        x = self.bn4(x, noise, training)
        x = self.rel4(x)
        x = self.conv4(x, mask)
        x = self.bn5(x, noise, training)
        x = self.rel5(x)
        x = self.conv5(x, mask)
        x = self.bn6(x, noise, training)
        x = self.rel6(x)
        x = self.conv6(x, mask)
        x = self.bn7(x, noise, training)
        x = self.rel7(x)
        x = self.conv7(x, mask)
        x = self.bn8(x, noise, training)
        x = self.rel8(x)
        x = self.conv8(x, mask)
        x = self.bn9(x, noise, training)
        x = self.rel9(x)
        x = self.conv9(x, mask)
        x = self.bn10(x, noise, training)
        x = self.rel10(x)
        x = self.conv10(x, mask)
        x = self.bn11(x, noise, training)
        x = self.rel11(x)
        x = self.conv11(x, mask)
        x = self.bn12(x, noise, training)
        x = self.rel12(x)
        x = self.conv12(x, mask)
        x = self.bn13(x, noise, training)
        x = self.rel13(x)
        x = self.conv13(x, mask)
        x = self.bn14(x, noise, training)
        x = self.rel14(x)
        x = self.conv14(x, mask)
        x = self.bn15(x, noise, training)
        x = self.rel15(x)
        x = self.conv15(x, mask)
        x = self.bn16(x, noise, training)
        x = self.rel16(x)
        x = self.conv16(x, mask)
        x = self.bn17(x, noise, training)
        x = self.rel17(x)
        x = self.conv17(x, mask)
        x = self.bn18(x, noise, training)
        x = self.rel18(x)
        x = self.conv18(x, mask)
        x = self.bn19(x, noise, training)
        x = self.rel19(x)
        x = self.conv19(x, mask)
        x = self.bn20(x, noise, training)
        x = self.rel20(x)
        x = self.conv20(x, mask)
        x = self.bn21(x, noise, training)
        x = self.rel21(x)
        x = self.conv21(x, mask)
        x = self.bn22(x, noise, training)
        x = self.rel22(x)
        x = self.conv22(x, mask)
        x = self.bn23(x, noise, training)
        x = self.rel23(x)
        x = self.conv23(x, mask)
        x = self.bn24(x, noise, training)
        x = self.rel24(x)
        x = self.conv24(x, mask)
        x = self.bn25(x, noise, training)
        x = self.rel25(x)
        x = self.conv25(x, mask)
        x = self.bn26(x, noise, training)
        x = self.rel26(x)
        x = self.conv26(x, mask)
        x = self.bn27(x, noise, training)
        x = self.rel27(x)
        x = self.conv27(x, mask)
        x = self.bn28(x, noise, training)
        x = self.rel28(x)
        x = self.conv28(x, mask)
        x = self.bn29(x, noise, training)
        x = self.rel29(x)
        x = self.conv29(x, mask)
        x = self.bn30(x, noise, training)
        x = self.rel30(x)
        x = self.conv30(x, mask)
        x = self.bn31(x, noise, training)
        x = self.rel31(x)
        x = self.conv31(x, mask)
        x = self.bn32(x, noise, training)
        x = self.rel32(x)
        x = self.conv32(x, mask)
        x = self.bn33(x, noise, training)
        x = self.rel33(x)
        x = self.conv33(x, mask)
        x = self.bn34(x, noise, training)
        x = self.rel34(x)
        x = self.conv34(x, mask)
        x = self.bn35(x, noise, training)
        x = self.rel35(x)
        x = self.conv35(x, mask)
        x = self.bn36(x, noise, training)
        x = self.rel36(x)
        x = self.conv36(x, mask)
        x = self.bn37(x, noise, training)
        x = self.rel37(x)
        x = self.conv37(x, mask)
        x = self.bn38(x, noise, training)
        x = self.rel38(x)
        x = self.conv38(x, mask)
        x = self.bn39(x, noise, training)
        x = self.rel39(x)
        x = self.conv39(x, mask)
        x = self.bn40(x, noise, training)
        x = self.rel40(x)
        x = self.conv40(x, mask)
        x = self.bn41(x, noise, training)
        x = self.rel41(x)
        x = self.conv41(x, mask)
        x = self.bn42(x, noise, training)
        x = self.rel42(x)
        x = self.conv42(x, mask)
        x = self.bn43(x, noise, training)
        x = self.rel43(x)
        x = self.conv43(x, mask)
        x = self.bn44(x, noise, training)
        x = self.rel44(x)
        x = self.conv44(x, mask)
        x = self.bn45(x, noise, training)
        x = self.rel45(x)
        x = self.conv45(x, mask)
        x = self.bn46(x, noise, training)
        x = self.rel46(x)
        x = self.conv46(x, mask)
        x = self.bn47(x, noise, training)
        x = self.rel47(x)
        x = self.conv47(x, mask)
        x = self.bn48(x, noise, training)
        x = self.rel48(x)
        x = self.conv48(x, mask)
        x = self.bn49(x, noise, training)
        x = self.rel49(x)
        x = self.conv49(x, mask)
        x = self.bn50(x, noise, training)
        x = self.rel50(x)
        x = self.conv50(x, mask)
        x = self.bn51(x, noise, training)
        x = self.rel51(x)
        x = self.conv51(x, mask)
        x = self.bn52(x, noise, training)
        x = self.rel52(x)
        x = self.conv52(x, mask)
        x = self.bn53(x, noise, training)
        x = self.rel53(x)
        x = self.conv53(x, mask)
        x = self.bn54(x, noise, training)
        x = self.rel54(x)
        x = self.conv54(x, mask)
        x = self.bn55(x, noise, training)
        x = self.rel55(x)
        x = self.conv55(x, mask)
        x = self.bn56(x, noise, training)
        x = self.rel56(x)
        x = self.conv56(x, mask)
        x = self.bn57(x, noise, training)
        x = self.rel57(x)
        x = self.conv57(x, mask)
        x = self.bn58(x, noise, training)
        x = self.rel58(x)
        x = self.conv58(x, mask)
        x = self.bn59(x, noise, training)
        x = self.rel59(x)
        x = self.conv59(x, mask)
        x = self.bn60(x, noise, training)
        x = self.rel60(x)
        x = self.conv60(x, mask)
        unaligned_features = x
        x = self.bn61(x, noise, training)
        x = self.rel61(x)
        x = self.conv61(x)
        x = self.bn62(x, noise, training)
        x = self.rel62(x)
        x = self.conv62(x)
        
        token_lengths = self.rel63(tf.squeeze(x)) # -> [N, 600]
        
        return token_lengths, unaligned_features

class GBlock(keras.layers.Layer):
    def __init__(self, in_channels, out_channels, upsample_factor):
        super(GBlock, self).__init__()
        self.bn1 =  ConditionalBatchNormSpect(in_channels)
        self.bn2 =  ConditionalBatchNormSpect(out_channels)
        self.bn3 =  ConditionalBatchNormSpect(out_channels)
        self.bn4 =  ConditionalBatchNormSpect(out_channels)
        self.rel1 = keras.layers.Activation('relu')
        self.rel2 = keras.layers.Activation('relu')
        self.rel3 = keras.layers.Activation('relu')
        self.rel4 = keras.layers.Activation('relu')
        self.upsamp1 = tfa.layers.SpectralNormalization(keras.layers.Conv1DTranspose(in_channels, kernel_size=upsample_factor*2, strides=upsample_factor, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))
        self.upsamp2 = tfa.layers.SpectralNormalization(keras.layers.Conv1DTranspose(in_channels, kernel_size=upsample_factor*2, strides=upsample_factor, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))
        self.conv1 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))
        self.conv2 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal()))
        self.conv3 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 1, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))
        self.conv4 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', dilation_rate=4, kernel_initializer=tf.keras.initializers.orthogonal()))
        self.conv5 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', dilation_rate=8, kernel_initializer=tf.keras.initializers.orthogonal()))
        self.add1 = keras.layers.Add()
        self.add2 = keras.layers.Add()

    def call(self, x, noise, training=False):
        x_skip = x
        x = self.bn1(x, noise, training)
        x = self.rel1(x)
        x = self.upsamp1(x)
        x = self.conv1(x)
        x = self.bn2(x, noise, training)
        x = self.rel2(x)
        x = self.conv2(x)

        x_skip = self.upsamp2(x_skip)
        x_skip = self.conv3(x_skip)
        
        x = self.add1([x,x_skip])

        x_skip2 = x

        x = self.bn3(x, noise, training)
        x = self.rel3(x)
        x = self.conv4(x)
        x = self.bn4(x, noise, training)
        x = self.rel4(x)
        x = self.conv5(x)

        x = self.add2([x,x_skip2])
        return x

class ConditionalBatchNorm(keras.layers.Layer):
    def __init__(self, channels):
        super(ConditionalBatchNorm, self).__init__()
        self.dense = keras.layers.Dense(channels*2, kernel_initializer=tf.keras.initializers.orthogonal())
        self.bn= keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, x, noise, training=False):
        x = self.bn(x, training)
        noise = self.dense(noise)
        gamma, beta = tf.split(noise, num_or_size_splits=2, axis=1)
        gamma = tf.expand_dims(gamma, 0)
        beta = tf.expand_dims(beta, 0)
        x = x * gamma + beta
        return x

class ConditionalBatchNormSpect(keras.layers.Layer):
    def __init__(self, channels):
        super(ConditionalBatchNormSpect, self).__init__()
        self.dense = tfa.layers.SpectralNormalization(keras.layers.Dense(channels*2, kernel_initializer=tf.keras.initializers.orthogonal()))
        self.bn= keras.layers.BatchNormalization(center=False, scale=False)

    def call(self, x, noise, training=False):
        x = self.bn(x, training)
        noise = self.dense(noise)
        gamma, beta = tf.split(noise, num_or_size_splits=2, axis=1)
        gamma = tf.expand_dims(gamma, 0)
        beta = tf.expand_dims(beta, 0)
        x = x * gamma + beta
        return x

def DBlock(x, out_channels, downsample_factor):
    x_skip = x
    x = keras.layers.AveragePooling1D( pool_size=1, strides=downsample_factor)(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 3, padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.orthogonal()))(x)

    x_skip = tfa.layers.SpectralNormalization(keras.layers.Conv1D(out_channels, 1, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x_skip)
    x_skip = keras.layers.AveragePooling1D( pool_size=1, strides=downsample_factor)(x_skip)

    x = keras.layers.Add()([x, x_skip])

    return x

def SpectDisResDownBlock(x, in_channels, out_channels):
    x_skip = x
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (1,1), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (3,3), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (3,3), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.AveragePooling2D()(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels, (1,1), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)

    x_skip = keras.layers.AveragePooling2D()(x_skip)

    if in_channels != out_channels:
        x_concat = x_skip
        x_skip = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels-in_channels, (1,1), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
        x_skip = keras.layers.Concatenate()([x_skip,x_concat])
    
    x = keras.layers.Add()([x, x_skip])

    return x

def SpectDisResBlock(x, out_channels):
    x_skip = x
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (1,1), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (3,3), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels//4, (3,3), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('relu')(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(out_channels, (1,1), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    
    x = keras.layers.Add()([x, x_skip])

    return x

def uDiscriminator(x, factors):
    x = tf.reshape(x, [tf.shape(x)[0], 240, factors[0]])
    x = DBlock(x, 64,  factors[1])
    x = DBlock(x, 128,  factors[2])
    x = DBlock(x, 256,  factors[3])
    x = DBlock(x, 256,  factors[4])
    x = DBlock(x, 256,  factors[5])
    x = tf.math.reduce_mean(x, axis=(1,2), keepdims=True)
    x = keras.layers.Flatten()(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Dense(1))(x)
    x = keras.layers.Activation('sigmoid')(x)
    return x

def SpectrogramDiscriminator():
    x_input = keras.layers.Input((48000,))
    x = tf.expand_dims(get_mel_spectrogram(x_input), -1)
    x = keras.layers.ZeroPadding2D((1,1))(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Conv2D(64, (3,3), padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = SpectDisResDownBlock(x, 64, 64*8)
    x = SpectDisResDownBlock(x, 64*8, 64*16)
    x = SpectDisResDownBlock(x, 64*16, 64*16)
    x = SpectDisResBlock(x, 64*16)
    x = keras.layers.Activation('relu')(x)
    x = tf.math.reduce_sum(x, axis=(1,2))
    x = keras.layers.Flatten()(x)
    x = tfa.layers.SpectralNormalization(keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.orthogonal()))(x)
    x = keras.layers.Activation('sigmoid')(x)
    model = keras.models.Model(inputs= x_input, outputs= x)
    return model

def Discriminators(x, batch_size):
    x1_index = tf.random.uniform([batch_size],0,48000-240, dtype=tf.dtypes.int32)
    x1_index = tf.map_fn(fn=lambda t: tf.range(t, t + 240), elems=x1_index)

    x2_index = tf.random.uniform([batch_size],0,48000-480, dtype=tf.dtypes.int32)
    x2_index = tf.map_fn(fn=lambda t: tf.range(t, t + 480), elems=x2_index)

    x3_index = tf.random.uniform([batch_size],0,48000-960, dtype=tf.dtypes.int32)
    x3_index = tf.map_fn(fn=lambda t: tf.range(t, t + 960), elems=x3_index)

    x4_index = tf.random.uniform([batch_size],0,48000-1920, dtype=tf.dtypes.int32)
    x4_index = tf.map_fn(fn=lambda t: tf.range(t, t + 1920), elems=x4_index)

    x5_index = tf.random.uniform([batch_size],0,48000-3600, dtype=tf.dtypes.int32)
    x5_index = tf.map_fn(fn=lambda t: tf.range(t, t + 3600), elems=x5_index)


    x1 = uDiscriminator(tf.gather(x,x1_index, axis=1, batch_dims=1), (1,1,5,3,1,1))
    x2 = uDiscriminator(tf.gather(x,x2_index, axis=1, batch_dims=1), (2,1,5,3,1,1))
    x3 = uDiscriminator(tf.gather(x,x3_index, axis=1, batch_dims=1), (4,1,5,3,1,1))
    x4 = uDiscriminator(tf.gather(x,x4_index, axis=1, batch_dims=1), (8,1,5,3,1,1))
    x5 = uDiscriminator(tf.gather(x,x5_index, axis=1, batch_dims=1), (15,1,2,2,1,1))

    x6 = SpectrogramDiscriminator(get_mel_spectrogram(x))

def Discriminator(factors):
    x_input = keras.layers.Input((48000,))
    x = uDiscriminator(x_input, factors)
    model = keras.models.Model(inputs= x_input, outputs= x)
    return model

def alignerloss(lengths, mel_true, mel_gen):
    train_length = tf.ones(lengths.shape[0])*400
    length_loss = tf.math.reduce_mean(((tf.math.square(train_length -lengths))/2)*0.1)
    dtw_loss = 0
    for mel_tr, mel_g in zip(mel_true, mel_gen):
        dtw_loss += spectrogram_dtw_error(mel_tr, mel_g)
    dtw_loss = dtw_loss / mel_true.shape[0]
    loss = length_loss + dtw_loss
    return loss

class GAN(keras.Model):
    def __init__(self, discriminator1, discriminator2, discriminator3, discriminator4, discriminator5, spectdiscriminator, aligner):
        super(GAN, self).__init__()

        self.loss_d_fn = keras.losses.Hinge()
        self.discriminator1 = discriminator1
        self.discriminator2 = discriminator2
        self.discriminator3 = discriminator3
        self.discriminator4 = discriminator4
        self.discriminator5 = discriminator5
        self.spectdiscriminator = spectdiscriminator
        self.aligner = aligner

        self.bn = ConditionalBatchNorm(256)
        self.rel = keras.layers.Activation('relu')
        self.conv1 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(768, 3, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))
        self.gblock1 = GBlock(768, 768, 1)
        self.gblock2 = GBlock(768, 768, 1)
        self.gblock3 = GBlock(768, 384, 2)
        self.gblock4 = GBlock(384, 384, 2)
        self.gblock5 = GBlock(384, 384, 2)
        self.gblock6 = GBlock(384, 192, 3)
        self.gblock7 = GBlock(192, 96, 5)
        self.conv2 = tfa.layers.SpectralNormalization(keras.layers.Conv1D(1, 3, padding='same', kernel_initializer=tf.keras.initializers.orthogonal()))

        self.tanh = keras.layers.Activation('tanh')

    def call(self, x, noise=None, training=False):
        x = self.bn(x, noise, training)
        x = self.rel(x)
        x = self.conv1(x)
        x = self.gblock1(x, noise, training)
        x = self.gblock2(x, noise, training)
        x = self.gblock3(x, noise, training)
        x = self.gblock4(x, noise, training)
        x = self.gblock5(x, noise, training)
        x = self.gblock6(x, noise, training)
        x = self.gblock7(x, noise, training)
        x = self.conv2(x)
        x = self.tanh(x)
        return x

    def compile(self, d_optimizer, g_optimizer, al_optimizer):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.al_optimizer = al_optimizer
        self.loss_d_fn = keras.losses.Hinge()
        self.loss_g_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.d1_loss_metric = keras.metrics.Mean(name="d1_loss")
        self.d2_loss_metric = keras.metrics.Mean(name="d2_loss")
        self.d3_loss_metric = keras.metrics.Mean(name="d3_loss")
        self.d4_loss_metric = keras.metrics.Mean(name="d4_loss")
        self.d5_loss_metric = keras.metrics.Mean(name="d5_loss")
        self.d6_loss_metric = keras.metrics.Mean(name="d6_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.al_loss_metric = keras.metrics.Mean(name="al_loss")

    @property
    def metrics(self):
        return [self.d1_loss_metric, self.d2_loss_metric, self.d3_loss_metric, self.d4_loss_metric, self.d5_loss_metric, self.d6_loss_metric, self.g_loss_metric, self.al_loss_metric]

    def train_step(self, data):
       
        phonemes, real_speech = data
        batch_size = 4

    
        speech_offset = real_speech[:,-1]
        real_speech = real_speech[:,:-1]


        lengths = phonemes[:,-1]
        phonemes = phonemes[:,:-1]
       
        phonemes_1 = phonemes[:batch_size]
        phonemes_2 = phonemes[batch_size:]

        lengths_1 = lengths[:batch_size]
        lengths_2 = lengths[batch_size:]

        real_speech_1 = real_speech[:batch_size]
        real_speech_2 = real_speech[batch_size:]

        real_speech_offset_1 = speech_offset[:batch_size]
        real_speech_offset_2 = speech_offset[batch_size:]

        noise = tf.random.normal(shape=(batch_size, 128))
        
        # Decode them to fake speech
        token_lenghts, unaligned_features = self.aligner(phonemes_1, noise)
        token_ends = tf.math.cumsum(token_lenghts, axis=1) # -> [N, 600]

        token_centres = token_ends - (token_lenghts / 2.) # -> [N, 600]
        # Compute predicted length as the last valid entry of token_ends. -> [N]
    
        # Compute output grid -> [N, out_sequence_length=6000]
        aligned_lengths = token_ends[0, lengths_1[0]-1]
        for i in range(1,batch_size):
            aligned_lengths = tf.concat([aligned_lengths, token_ends[i, lengths_1[i]-1]], 0)

        out_pos = tf.stack([tf.range(400, dtype=tf.float32) + real_speech_offset_1[0]])
        for i in range(1,batch_size):
            out_pos = tf.concat([out_pos, tf.stack([tf.range(400, dtype=tf.float32) + real_speech_offset_1[i]])], 0)

        out_pos = tf.expand_dims(out_pos, -1) # -> [N, 6000, 1]
        diff = tf.expand_dims(token_centres, 1) - out_pos # -> [N, 6000, 600]
        logits = -(diff**2 / 10.) # -> [N, 6000, 600]
        
        # Mask out invalid input locations (flip 0/1 to 1/0); add dummy output axis.
        sequence_length = 400
        logits_inv_mask = tf.expand_dims(tf.stack([tf.where(tf.range(sequence_length) < tf.cast(lengths_1[0], dtype=tf.int32) , x=0., y=1.)]), 1) # -> [N, 1, 600]
        for i in range(1,batch_size):
            logits_inv_mask = tf.concat([logits_inv_mask, tf.expand_dims(tf.stack([tf.where(tf.range(sequence_length) < tf.cast(lengths_1[i], dtype=tf.int32), x=0., y=1.)]), 1)], 0) # -> [N, 1, 600]

        masked_logits = logits - 1e9 * logits_inv_mask # -> [N, 6000, 600]

        weights = self.softmax(masked_logits) # -> [N, 6000, 600]
        # Do a batch matmul (written as an einsum) to compute the aligned features.
        # aligned_features -> [N, 6000, 256]
        aligned_features = tf.einsum('noi,nid->nod', weights, unaligned_features)

        generated_speech = self(aligned_features, noise, out_offset=real_speech_offset_1, train_sequence_length=True)

        # Combine them with real speech
        combined_speech = tf.concat([generated_speech, real_speech_1], axis=0)

        # Assemble labels discriminating real from fake speech
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        
        # Train the discriminators
        x1_index = tf.random.uniform([batch_size*2],0,48000-240, dtype=tf.dtypes.int32)
        x1_index = tf.map_fn(fn=lambda t: tf.range(t, t + 240), elems=x1_index)

        x2_index = tf.random.uniform([batch_size*2],0,48000-480, dtype=tf.dtypes.int32)
        x2_index = tf.map_fn(fn=lambda t: tf.range(t, t + 480), elems=x2_index)

        x3_index = tf.random.uniform([batch_size*2],0,48000-960, dtype=tf.dtypes.int32)
        x3_index = tf.map_fn(fn=lambda t: tf.range(t, t + 960), elems=x3_index)

        x4_index = tf.random.uniform([batch_size*2],0,48000-1920, dtype=tf.dtypes.int32)
        x4_index = tf.map_fn(fn=lambda t: tf.range(t, t + 1920), elems=x4_index)

        x5_index = tf.random.uniform([batch_size*2],0,48000-3600, dtype=tf.dtypes.int32)
        x5_index = tf.map_fn(fn=lambda t: tf.range(t, t + 3600), elems=x5_index)

        with tf.GradientTape() as tape:
            predictions = self.discriminator1(tf.gather(combined_speech, x1_index, axis=1, batch_dims=1))
            d1_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d1_loss, self.discriminator1.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator1.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator2(tf.gather(combined_speech, x2_index, axis=1, batch_dims=1))
            d2_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d2_loss, self.discriminator2.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator2.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator3(tf.gather(combined_speech, x3_index, axis=1, batch_dims=1))
            d3_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d3_loss, self.discriminator3.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator3.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator4(tf.gather(combined_speech, x4_index, axis=1, batch_dims=1))
            d4_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d4_loss, self.discriminator4.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator4.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.discriminator5(tf.gather(combined_speech, x5_index, axis=1, batch_dims=1))
            d5_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d5_loss, self.discriminator5.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator5.trainable_weights)
        )

        with tf.GradientTape() as tape:
            predictions = self.spectdiscriminator(combined_speech)
            d6_loss = self.loss_d_fn(labels, predictions)
        grads = tape.gradient(d6_loss, self.spectdiscriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.spectdiscriminator.trainable_weights)
        )

        # Assemble labels that say "all real speech"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!


        x1_index = tf.random.uniform([batch_size],0,48000-240, dtype=tf.dtypes.int32)
        x1_index = tf.map_fn(fn=lambda t: tf.range(t, t + 240), elems=x1_index)

        x2_index = tf.random.uniform([batch_size],0,48000-480, dtype=tf.dtypes.int32)
        x2_index = tf.map_fn(fn=lambda t: tf.range(t, t + 480), elems=x2_index)

        x3_index = tf.random.uniform([batch_size],0,48000-960, dtype=tf.dtypes.int32)
        x3_index = tf.map_fn(fn=lambda t: tf.range(t, t + 960), elems=x3_index)

        x4_index = tf.random.uniform([batch_size],0,48000-1920, dtype=tf.dtypes.int32)
        x4_index = tf.map_fn(fn=lambda t: tf.range(t, t + 1920), elems=x4_index)

        x5_index = tf.random.uniform([batch_size],0,48000-3600, dtype=tf.dtypes.int32)
        x5_index = tf.map_fn(fn=lambda t: tf.range(t, t + 3600), elems=x5_index)

        noise = tf.random.normal(shape=(batch_size, 128))


        with tf.GradientTape() as tape:
            token_lenghts, unaligned_features = self.aligner(phonemes_2, noise)
            token_ends = tf.math.cumsum(token_lenghts, axis=1) # -> [N, 600]

            token_centres = token_ends - (token_lenghts / 2.) # -> [N, 600]
            # Compute predicted length as the last valid entry of token_ends. -> [N]
        
            # Compute output grid -> [N, out_sequence_length=6000]
            aligned_lengths = token_ends[0, lengths_2[0]-1]
            for i in range(1,batch_size):
                aligned_lengths = tf.concat([aligned_lengths, token_ends[i, lengths_2[i]-1]], 0)

            out_pos = tf.stack([tf.range(400, dtype=tf.float32) + real_speech_offset_2[0]])
            for i in range(1,batch_size):
                out_pos = tf.concat([out_pos, tf.stack([tf.range(400, dtype=tf.float32) + real_speech_offset_2[i]])], 0)

            out_pos = tf.expand_dims(out_pos, -1) # -> [N, 6000, 1]
            diff = tf.expand_dims(token_centres, 1) - out_pos # -> [N, 6000, 600]
            logits = -(diff**2 / 10.) # -> [N, 6000, 600]
            # Mask out invalid input locations (flip 0/1 to 1/0); add dummy output axis.
            sequence_length = 400
            logits_inv_mask = tf.expand_dims(tf.stack([tf.where(tf.range(sequence_length) < tf.cast(lengths_2[0], dtype=tf.int32) , x=0., y=1.)]), 1) # -> [N, 1, 600]
            for i in range(1,batch_size):
                logits_inv_mask = tf.concat([logits_inv_mask, tf.expand_dims(tf.stack([tf.where(tf.range(sequence_length) < tf.cast(lengths_2[i], dtype=tf.int32), x=0., y=1.)]), 1)], 0) # -> [N, 1, 600]

            masked_logits = logits - 1e9 * logits_inv_mask # -> [N, 6000, 600]

            weights = keras.activations.softmax(masked_logits) # -> [N, 6000, 600]
            # Do a batch matmul (written as an einsum) to compute the aligned features.
            # aligned_features -> [N, 6000, 256]
            aligned_features = tf.einsum('noi,nid->nod', weights, unaligned_features)

            mel_true = get_mel_spectrogram(real_speech_2, jitter=True)
            mel_gen =  get_mel_spectrogram(self(aligned_features, noise))
            al_loss = alignerloss(aligned_lengths, mel_true, mel_gen)
        
        grads = tape.gradient(al_loss, self.aligner.trainable_weights)
        self.al_optimizer.apply_gradients(
            zip(grads, self.aligner.trainable_weights)
        )


        with tf.GradientTape() as tape:
            generated_speech = self(aligned_features, noise, training=True)
            predictions1 = self.discriminator1(tf.gather(generated_speech, x1_index, axis=1, batch_dims=1))
            predictions2 = self.discriminator2(tf.gather(generated_speech, x2_index, axis=1, batch_dims=1))
            predictions3 = self.discriminator3(tf.gather(generated_speech, x3_index, axis=1, batch_dims=1))
            predictions4 = self.discriminator4(tf.gather(generated_speech, x4_index, axis=1, batch_dims=1))
            predictions5 = self.discriminator5(tf.gather(generated_speech, x5_index, axis=1, batch_dims=1))
            predictions6 = self.spectdiscriminator(generated_speech)
            predictions, _ = stats.mode(tf.stack(predictions1,predictions2,predictions3,predictions4,predictions5,predictions6).numpy())
            predictions = tf.convert_to_tensor(predictions)
            
            g_loss = self.loss_g_fn(misleading_labels,predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d1_loss_metric.update_state(d1_loss)
        self.d2_loss_metric.update_state(d2_loss)
        self.d3_loss_metric.update_state(d3_loss)
        self.d4_loss_metric.update_state(d4_loss)
        self.d5_loss_metric.update_state(d5_loss)
        self.d6_loss_metric.update_state(d6_loss)
        self.g_loss_metric.update_state(g_loss)
        self.al_loss_metric.update_state(al_loss)
        return {
            "d1_loss": self.d1_loss_metric.result(),
            "d2_loss": self.d2_loss_metric.result(),
            "d3_loss": self.d3_loss_metric.result(),
            "d4_loss": self.d4_loss_metric.result(),
            "d5_loss": self.d5_loss_metric.result(),
            "d6_loss": self.d6_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
            "al_loss": self.g_loss_metric.result(),
        }
