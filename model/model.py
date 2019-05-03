from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import warnings
import tensorflow as tf
from keras.applications import imagenet_utils
from keras.layers import Activation
from keras.layers import Input

def relu6(x):
    return tf.nn.relu6(x)

#Error - keras module
def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')

#Insight - understand difference between kernel and filter
def _conv_block(inputs, filters, alpha, kernel=(3,3), strides=(1,1)):
    channel_axis = -1
    filters = int(filters * alpha)
    x = tf.pad(inputs, paddings=(1,1), name='conv1_pad')
    x = tf.layers.conv2d(x, filters, kernel, padding='valid', use_bias=False, strides=strides, name='conv1')
    x = tf.layers.batch_normalization(x, axis=channel_axis, name='conv1_bn')
    //keras module
    return Activation(relu6(x), name='conv1_relu')

def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1,1), block_id=1):
    channel_axis = -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = tf.pad(inputs, paddings=(1,1), name='conv_pad' % block_id)
    #Error - no name in this layer, need to pass input better
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False)(x)
    x = tf.layers.batch_normalization(x, axis=channel_axis, name='conv_dw_%d_bn' % block_id)
    return Activation(relu6(x), name='conv_dw_%d_relu' %block_id)

def _depthwise_conv_block_f(inputs, depth_multiplier=1, strides=(1,1), strides=(1,1), block_id=1):
    channel_axis = -1
    x = tf.pad(inputs, paddings=(1,1), name='conv_pad_%d' % block_id)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,3), padding='valid', depth_multiplier=depth_multiplier, strides=strides, use_bias=False)(x)
    x = tf.layers.batch_normalization(x, axis=channel_axis, name='conv_dw_%d_bn' % block_id)
    return Activation(relu6(x), name='conv_pw_%d_relu' % block_id)


def _conv_blockSSD_f(inputs, filters, alpha, kernel, strides, block_id=11):
    channel_axis = -1
    filters = int(filters * alpha)
    Conv = tf.layers.conv2d(inputs, filters, kernel, padding='valid', use_bias='False', strides=strides, name='conv_%d' % block_id)
    x = tf.layers.batch_normalization(Conv, axis=channel_axis, name='conv_%_bn' % block_id)
    return Activation(relu6(x), name='conv_%d_relu' % block_id), Conv

def _conv_blockSSD(inputs, filters, alpha, block_id=11):
    channel_axis = -1
    filters = int(filters * alpha)
    x = tf.pad(inputs, paddings=(1,1), name='conv_pad_#d_1' % block_id)
    #Error - no kernel_size defined
    x = tf.layers.conv2d(x, filters, padding='valid', use_bias=False, strides=(1,1), name='conv_%d_1' % block_id)
    x = tf.layers.batch_normalization(x, axis=channel_axis, name='conv_%d_bn_1' % block_id)
    x = Activation(relu6(x), name='conv_%d_relu_1' % block_id)
    conv = tf.layers.conv2d(x, filters*2, kernel_size=(3,3), padding='valid', use_bias=False, strides=(2,2), name='conv_%d_2' % block_id)
    x = tf.layers.batch_normalization(conv, axis=channel_axis, name='conv_%d_bn_2' % block_id)
    x = Activation(relu6(x), name='conv_%d_relu_2' % block_id)
    return x, conv

def SSD(input_shape, num_classes):
    img_size = (input_shape[1], input_shape[0])
    input_shape = (input_shape[1], input_shape[0], 3)
    alpha = 1.0
    depth_multiplier = 1
    input0 = Input(input_shape)
    x = _conv_block(input0, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block_f(x, depth_multiplier, strides=(1, 1), block_id=11)
    x, conv11 = _conv_blockSSD_f(x, 512, depth_multiplier, kernel=(1, 1), strides=(1, 1), block_id=11)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block_f(x, depth_multiplier, strides=(1, 1), block_id=13)
    x, conv13 = _conv_blockSSD_f(x, 512, alpha, kernel=(1, 1), strides=(1, 1), block_id=13)
    x, conv14_2 = _conv_blockSSD(x, 256, alpha, block_id=14)
    x, conv15_2 = _conv_blockSSD(x, 128, alpha, block_id=15)
    x, conv16_2 = _conv_blockSSD(x, 128, alpha, block_id=16)
    x, conv17_2 = _conv_blockSSD(x, 64, alpha, block_id=17)

    #Prediction from conv11
    num_priors = 3
    x = tf.layers.conv2d(conv11, num_priors * 4, (1, 1), padding='same', name='conv11_mbox_loc')
    conv11_mbox_loc = x
    conv11_mbox_loc_flat = tf.layers.flatten(conv11_mbox_loc, name='conv11_mbox_loc')
    name = 'conv11_mbox_conf'
    conv11_mbox_conf = tf.layers.conv2d(conv11, num_priors * num_classes, (1, 1), padding='same', name=name)
    conv11_mbox_conf_flat = tf.layers.flatten(conv11_mbox_conf, name='conv11_mbox_conf_flat')
    priorbox  = PriorBox(img_size, 60, max_size=None, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv11_mbox_priorbox')
    conv11_mbox_priorbox = priorbox(conv11)

    num_priors = 6
    x = tf.layers.conv2d(conv13, num_priors * 4, (1, 1), padding='same', name='conv13_mbox_loc')(conv13)
    conv13_mbox_loc = x
    flatten = tf.layers.flatten()



        



