import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
import scipy.io
import matplotlib.cm as cm
from functools import partial


def save_image(image, save_dir, name):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)

        
def add_image_summary(var,**kwargs):
    if var is None:
        return
    num_ch = int(var.get_shape()[-1])
    var_ch_lst = tf.split(var, num_or_size_splits=num_ch, axis=3)
    for ch, var_ch in enumerate(var_ch_lst):
        tf.summary.image(var.op.name + "/image_ch%d"%ch, var_ch, **kwargs)

        
def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    src="https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b.js"
    
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))
    
    # gather
    c_map = cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = c_map(np.arange(256))[:, :3]
    colors = tf.constant(colors, dtype=tf.float32)
    value = tf.gather(colors, indices)

    return value
    
    
def batch_colorize(tensor, vmin=None, vmax=None, cmap=None):
    tensor = tf.cast(tensor, tf.float32)
    colorize_tensor = partial(colorize, vmin=vmin, vmax=vmax, cmap=cmap)
    colorized_tensor = tf.map_fn(colorize_tensor, tensor)
    return colorized_tensor
    
    
def batch_colorize_ndarray(value, vmin=None, vmax=None, cmap=None):
    # normalize
    vmin = np.min(value[:]) if vmin is None else vmin
    vmax = np.max(value[:]) if vmax is None else vmax
    value = 255 * (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = np.squeeze(value)
    
    # gather
    c_map = cm.get_cmap(cmap if cmap is not None else 'gray')
    value = 255 * c_map(value.astype(np.int32))
    value = value.astype(np.uint8)
    return value
    
    
