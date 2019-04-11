import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import math_ops
from lib.utils import rnn_cell
import numpy as np
from util import log


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x > 0.0, x, alpha * tf.exp(x) - alpha)


def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


def instance_norm(input, is_training):
    """ instance normalization """
    with tf.variable_scope('instance_norm'):
        num_out = input.get_shape()[-1]
        scale = tf.get_variable(
            'scale', [num_out],
            initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        offset = tf.get_variable(
            'offset', [num_out],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
        mean, var = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-6
        inv = tf.rsqrt(var + epsilon)
        return scale * (input - mean) * inv + offset


def bn_act(input, is_train, norm='batch', activation_fn=None, name="bn_act"):
    with tf.variable_scope(name):
        _ = input
        if activation_fn is not None:
            _ = activation_fn(_)
        if norm is not None and norm is not False:
            if norm == 'batch':
                # batch norm
                _ = tf.contrib.layers.batch_norm(
                    _, center=True, scale=True, decay=0.999,
                    is_training=is_train, updates_collections=None
                )
            elif norm == 'instance':
                _ = instance_norm(_, is_train)
            elif norm == 'None':
                _ = _
    return _


def conv2d(input, output_shape, is_train, info=False, k_h=4, k_w=4, s=2,
           stddev=0.01, name="conv2d", activation_fn=lrelu,
           norm='batch'):
    with tf.variable_scope(name):
        _ = slim.conv2d(input, output_shape, [k_h, k_w], stride=s, activation_fn=None)
        _ = bn_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _


def residual_block(input, output_shape, is_train, info=False, k=3, s=1,
                   name="residual", activation_fn=lrelu, norm='batch'):
    with tf.variable_scope(name):
        with tf.variable_scope('res1'):
            _ = conv2d(input, output_shape, is_train, k_h=k, k_w=k, s=s,
                       activation_fn=activation_fn, norm=norm)
        with tf.variable_scope('res2'):
            _ = conv2d(input, output_shape, is_train, k_h=k, k_w=k, s=s,
                       activation_fn=None, norm=norm)
        _ = activation_fn(_ + input)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _


def deconv2d(input, deconv_info, is_train, name="deconv2d", info=False,
             stddev=0.01, activation_fn=tf.nn.relu, norm='batch'):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        _ = layers.conv2d_transpose(
            input,
            num_outputs=output_shape,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.zeros_initializer(),
            kernel_size=[k, k], stride=[s, s], padding='SAME'
        )
        _ = bn_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _


def bilinear_deconv2d(input, deconv_info, is_train, name="bilinear_deconv2d",
                      info=False, activation_fn=tf.nn.relu, norm='batch'):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_bilinear(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k_h=k, k_w=k, s=1,
                   norm=False, activation_fn=None)
        _ = bn_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _


def nn_deconv2d(input, deconv_info, is_train, name="nn_deconv2d",
                info=False, activation_fn=tf.nn.relu, norm='batch'):
    with tf.variable_scope(name):
        output_shape = deconv_info[0]
        k = deconv_info[1]
        s = deconv_info[2]
        h = int(input.get_shape()[1]) * s
        w = int(input.get_shape()[2]) * s
        _ = tf.image.resize_nearest_neighbor(input, [h, w])
        _ = conv2d(_, output_shape, is_train, k_h=k, k_w=k, s=1,
                   norm=False, activation_fn=None)
        _ = bn_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _.get_shape().as_list()))
    return _


def residual_conv(input, num_filters, filter_size, stride, reuse=False,
                  pad='SAME', dtype=tf.float32, bias=False, name='res_conv'):
    with tf.variable_scope(name):
        stride_shape = [1, stride, stride, 1]
        filter_shape = [filter_size, filter_size, input.get_shape()[3], num_filters]
        w = tf.get_variable('w', filter_shape, dtype, tf.random_normal_initializer(0.0, 0.02))
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    return conv


def residual(input, num_filters, name, is_train, reuse=False, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = residual_conv(input, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )
            out = tf.nn.relu(out)

        with tf.variable_scope('res2', reuse=reuse):
            out = residual_conv(out, num_filters, 3, 1, reuse, pad, name=name)
            out = tf.contrib.layers.norm(
                out, center=True, scale=True, decay=0.9,
                is_training=is_train, updates_collections=None
            )

        return tf.nn.relu(input + out)


def fc(input, output_shape, is_train, info=False, norm='batch',
       activation_fn=lrelu, name="fc"):
    with tf.variable_scope(name):
        _ = slim.fully_connected(input, output_shape, activation_fn=None)
        _ = bn_act(_, is_train, norm=norm, activation_fn=activation_fn)
        if info: log.info('{} {}'.format(name, _))
    return _


def add_image_coord(img):
    h, w = int(img.get_shape()[1]), int(img.get_shape()[2])
    bs = int(img.get_shape()[0])
    x_coord = tf.convert_to_tensor(
        np.tile(np.expand_dims(np.expand_dims(np.tile(
            np.expand_dims(2 * np.array(range(w))/w - 1, axis=0), [h, 1]),
            axis=-1), axis=0), [bs, 1, 1, 1]),
        dtype=tf.float32
    )
    y_coord = tf.convert_to_tensor(
        np.tile(np.expand_dims(np.expand_dims(np.tile(
            np.expand_dims(2 * np.array(range(h))/h - 1, axis=1), [1, w]), axis=-1),
            axis=0), [bs, 1, 1, 1]),
        dtype=tf.float32
    )
    return tf.concat([img, x_coord, y_coord], axis=-1)


def get_pixel_value(img, x, y):
    """
    This code is from STN: https://github.com/kevinzakka/spatial-transformer-network
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W, )
    - y: flattened tensor of shape (B*H*W, )
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, x, y, background=False):
    """
    This code is from STN: https://github.com/kevinzakka/spatial-transformer-network
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - interpolated images according to grids. Same size as grid.
    """
    # prepare useful params
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]

    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # cast indices as float32 (for rescaling)
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    # rescale x and y to [0, W/H]
    x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    # i.e. we need a rectangle around the point of interest
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H/W] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out


def convlstm(input, state, activation=math_ops.sigmoid, kernel_shape=[3, 3],
             norm='batch', is_training=True, reuse=False, name='convlstm'):
    with tf.variable_scope(name, reuse=reuse):
        output_size = input.get_shape().as_list()[-1]
        cell = rnn_cell.ConvLSTMCell(conv_ndims=2,
                                     input_shape=input.get_shape().as_list()[1:],
                                     output_channels=output_size,
                                     kernel_shape=kernel_shape,
                                     skip_connection=False,
                                     initializers=tf.truncated_normal_initializer(stddev=0.02),
                                     activation=activation,
                                     name=name)

        if state is None:
            state = cell.zero_state(input.get_shape().as_list()[0], input.dtype)
        output, new_state = cell(input, state)

        output = bn_act(output, is_training, norm=norm, activation_fn=None)
    return output, new_state


def resnet_block_convlstm(input, state, activation=None, norm='batch',
                          is_training=True, reuse=False, name='res_block_convlstm'):
    with tf.variable_scope(name, reuse=reuse):
        assert len(state) == 2

        output, state1 = convlstm(input, state[0], activation=math_ops.sigmoid,
                                  norm=norm,
                                  is_training=True, reuse=reuse, name=name + '_1')
        output, state2 = convlstm(output, state[1], activation=math_ops.sigmoid,
                                  norm=norm,
                                  is_training=True, reuse=reuse, name=name + '_2')

    output = activation(output + input) if activation else output + input
    return output, (state1, state2)
