# -*- coding: utf-8 -*

import tensorflow as tf
import custom_layers

slim = tf.contrib.slim


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.get_variable(name, initializer=initial)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.02)
    return tf.get_variable(name, initializer=initial)

def conv2d_transpose(net, output_shape, kernel_size, scope):
    with tf.variable_scope(scope):
        in_channel = net.get_shape()[-1].value
        w = weight_variable(shape = kernel_size + [output_shape[-1]] + [in_channel],
                            name  = 'weight')
        b = bias_variable(shape = [output_shape[-1]], name='bias')
        conv = tf.nn.conv2d_transpose(net, w,
                                      output_shape = output_shape,
                                      strides      = [1, 2, 2, 1],
                                      padding      = "SAME")
        return tf.nn.bias_add(conv, b)

def deconv_module(skip, net, output_hw, output_channel, is_training):
    batch_norm_params = {'is_training': is_training,
                         'zero_debias_moving_mean': True,
                         'decay': 0.999,
                         'epsilon': 0.001,
                         'scale': False,
                         'updates_collections': tf.GraphKeys.UPDATE_OPS}

    skip = slim.conv2d(skip, 256, [3, 3], scope='conv_1', normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params)
    skip = slim.conv2d(skip, 256, [3, 3], scope='conv_2', normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params)
    # net = skip + tf.image.resize_images(net, output_shape)
    batch_size = net.get_shape()[0].value
    net  = conv2d_transpose(net,
                            output_shape=[batch_size] + output_hw + [output_channel],
                            kernel_size =[2, 2],
                            scope       ='deconv')
    net  = slim.conv2d(net, 256, [3, 3], scope='conv', normalizer_fn=slim.batch_norm,
                       normalizer_params=batch_norm_params)
    net  = net + skip
    return net

def basenet(inputs, fatness = 64, dilation = True, is_training = True):
    """
    backbone net of vgg16 + DSSD
    """
    # End_points collect relevant activations for external use.
    end_points = {}
    with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
        # Block1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['conv1_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        end_points['pool1'] = net

        # Block 2.
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['conv2_2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        end_points['pool2'] = net
        
        # Block 3.
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['conv3_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        end_points['pool3'] = net
        
        # Block 4.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')     # slim.conv2d stride默认=1
        end_points['conv4_3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')                   # slim.max_pool2d stride默认=2
        end_points['pool4'] = net
        
        # Block 5.
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['conv5_3'] = net
        net = slim.max_pool2d(net, [2, 2], 1, scope='pool5')
        end_points['pool5'] = net

        # fc6 as conv, dilation is added
        if dilation:
            net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='fc6')
        else:
            net = slim.conv2d(net, 1024, [3, 3], scope='fc6')
        end_points['fc6'] = net
        net = tf.layers.dropout(net, rate=0.5, training=is_training)

        # fc7 as conv
        net = slim.conv2d(net, 1024, [1, 1], scope='fc7')
        end_points['fc7'] = net
        net = tf.layers.dropout(net, rate=0.5, training=is_training)

        # conv8
        end_point = 'conv8'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # conv9
        end_point = 'conv9'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # conv10
        end_point = 'conv10'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net

        # conv11
        end_point = 'F11'
        with tf.variable_scope(end_point):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points[end_point] = net
        base_hw = [net.get_shape()[1].value, net.get_shape()[2].value]  ######################################

        # deconvolution module
        end_point = 'F10'
        with tf.variable_scope(end_point):
            skip = end_points['conv10']
            net = deconv_module(skip, net, [base_hw[0]*2, base_hw[1]*2], 256, is_training)
        end_points[end_point] = net

        end_point = 'F9'
        with tf.variable_scope(end_point):
            skip = end_points['conv9']
            net = deconv_module(skip, net, [base_hw[0]*4, base_hw[1]*4], 256, is_training)
        end_points[end_point] = net

        end_point = 'F8'
        with tf.variable_scope(end_point):
            skip = end_points['conv8']
            net = deconv_module(skip, net, [base_hw[0]*8, base_hw[1]*8], 256, is_training)
        end_points[end_point] = net

        end_point = 'F7'
        with tf.variable_scope(end_point):
            skip = end_points['fc7']
            net = deconv_module(skip, net, [base_hw[0]*16, base_hw[1]*16], 256, is_training)
        end_points[end_point] = net

        end_point = 'F4'
        with tf.variable_scope(end_point):
            skip = end_points['conv4_3']
            net = deconv_module(skip, net, [base_hw[0]*32, base_hw[1]*32], 256, is_training)
        end_points[end_point] = net

        end_point = 'F3'
        with tf.variable_scope(end_point):
            skip = end_points['conv3_3']
            net = deconv_module(skip, net, [base_hw[0]*64, base_hw[1]*64], 256, is_training)
        end_points[end_point] = net

    return net, end_points

