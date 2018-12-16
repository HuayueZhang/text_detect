# -*- coding: utf-8 -*
import math

import numpy as np
import tensorflow as tf
import ssd_common

import tf_extended_ssd as tfe
from nets import custom_layers

slim = tf.contrib.slim

import config


# =========================================================================== #
# SSD class definition.
# =========================================================================== #


class SSDNet(object):
    """Implementation of the SSD VGG-based 300 network.

    The default features layers with 300x300 image input are:
      conv4 ==> 38 x 38
      conv7 ==> 19 x 19
      conv8 ==> 10 x 10
      conv9 ==> 5 x 5
      conv10 ==> 3 x 3
      conv11 ==> 1 x 1
    The default image size used to train this network is 300x300.
    """

    # ======================================================================= #


    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        """Network arg_scope.
        """
        return ssd_arg_scope(weight_decay, data_format=data_format)

    def arg_scope_caffe(self, caffe_scope):
        """Caffe arg_scope used for weights importing.
        """
        return ssd_arg_scope_caffe(caffe_scope)

    # ======================================================================= #


    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return ssd_anchors_all_layers(img_shape,
                                      config.feat_shapes,
                                      config.anchor_sizes,
                                      config.anchor_ratios,
                                      config.anchor_steps,
                                      config.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors,
                      scope=None):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_encode(
            labels, bboxes, anchors,
            config.num_classes,
            config.no_annotation_label,
            ignore_threshold=0.5,
            prior_scaling=config.prior_scaling,
            scope=scope)

    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=config.prior_scaling,
            scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):
        """Get the detected bounding boxes from the SSD network output.
        """
        # Select top_k bboxes from predictions, and clip
        rscores, rbboxes = \
            ssd_common.tf_ssd_bboxes_select(predictions, localisations,
                                            select_threshold=select_threshold,
                                            num_classes=config.num_classes)
        rscores, rbboxes = \
            tfe.bboxes_sort(rscores, rbboxes, top_k=top_k)
        # Apply NMS algorithm.
        rscores, rbboxes = \
            tfe.bboxes_nms_batch(rscores, rbboxes,
                                 nms_threshold=nms_threshold,
                                 keep_top_k=keep_top_k)
        if clipping_bbox is not None:
            rbboxes = tfe.bboxes_clip(clipping_bbox, rbboxes)
        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """Define the SSD network losses.
        """
        return ssd_losses(logits, localisations,
                          gclasses, glocalisations, gscores,
                          match_threshold=match_threshold,
                          negative_ratio=negative_ratio,
                          alpha=alpha,
                          label_smoothing=label_smoothing,
                          scope=scope)


# =========================================================================== #
# SSD tools...
# =========================================================================== #
def ssd_size_bounds_to_values(size_bounds, n_feat_layers, img_shape=(300, 300)):
    """Compute the reference sizes of the anchor boxes from relative bounds.
    The absolute values are measured in pixels, based on the network
    default size (300 pixels).

    This function follows the computation performed in the original
    implementation of SSD in Caffe.

    Return:
      list of list containing the absolute sizes at each scale. For each scale,
      the ratios only apply to the first value.
    """
    assert img_shape[0] == img_shape[1]

    img_size = img_shape[0]
    min_ratio = int(size_bounds[0] * 100)
    max_ratio = int(size_bounds[1] * 100)
    step = int(math.floor((max_ratio - min_ratio) / (n_feat_layers - 2)))
    # Start with the following smallest sizes.
    sizes = [[img_size * size_bounds[0] / 2, img_size * size_bounds[0]]]
    for ratio in range(min_ratio, max_ratio + 1, step):
        sizes.append((img_size * ratio / 100.,
                      img_size * (ratio + step) / 100.))
    return sizes


def ssd_feat_shapes_from_net(predictions, default_shapes=None):
    """Try to obtain the feature shapes from the prediction layers. The latter
    can be either a Tensor or Numpy ndarray.

    Return:
      list of feature shapes. Default values if predictions shape not fully
      determined.
    """
    feat_shapes = []
    for l in predictions:
        # Get the shape, from either a np array or a tensor.
        if isinstance(l, np.ndarray):
            shape = l.shape
        else:
            shape = l.get_shape().as_list()
        shape = shape[1:4]
        # Problem: undetermined shape...
        if None in shape:
            return default_shapes
        else:
            feat_shapes.append(shape)
    return feat_shapes


def ssd_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         ratios,
                         step,
                         offset=0.5,
                         dtype=np.float32):
    """Computer SSD default anchor boxes for one feature layer.

    Determine the relative position grid of the centers, and the relative
    width and height.

    Arguments:
      feat_shape: Feature shape, used for computing relative position grids;
      size: Absolute reference sizes;
      ratios: Ratios to use on these features;
      img_shape: Image shape, used for computing height, width relatively to the
        former;
      offset: Grid offset.

    Return:
      y, x, h, w: Relative x and y grids, and height and width.
    """
    # Compute the position grid: simple way.
    # y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    # y = (y.astype(dtype) + offset) / feat_shape[0]
    # x = (x.astype(dtype) + offset) / feat_shape[1]
    # Weird SSD-Caffe computation using steps values...
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(sizes)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    for i, size in enumerate(sizes):
        h[i] = size / img_shape[0] / math.sqrt(ratios[0])
        w[i] = size / img_shape[1] * math.sqrt(ratios[0])
    return y, x, h, w


def ssd_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_ratios,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    for i, s in enumerate(layers_shape):
        anchor_bboxes = ssd_anchor_one_layer(img_shape, s,
                                             anchor_sizes[i],
                                             anchor_ratios[i],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


# =========================================================================== #
# Functional definition of VGG-based SSD 300.
# =========================================================================== #
def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([custom_layers.pad2d,
                                 custom_layers.l2_normalization,
                                 custom_layers.channel_to_last],
                                data_format=data_format) as sc:
                return sc


# =========================================================================== #
# Caffe scope: importing weights at initialization.
# =========================================================================== #
def ssd_arg_scope_caffe(caffe_scope):
    """Caffe scope definition.

    Args:
      caffe_scope: Caffe scope object with loaded weights.

    Returns:
      An arg_scope.
    """
    # Default network arg scope.
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=caffe_scope.conv_weights_init(),
                        biases_initializer=caffe_scope.conv_biases_init()):
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu):
            with slim.arg_scope([custom_layers.l2_normalization],
                                scale_initializer=caffe_scope.l2_norm_scale_init()):
                with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                    padding='SAME') as sc:
                    return sc


# =========================================================================== #
# SSD loss function.
# =========================================================================== #
def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/cpu:0',
               scope=None):

    lshape = tfe.get_shape(logits[0], 5)
    num_classes = lshape[-1]
    batch_size = lshape[0]

    # Flatten out all vectors!
    flogits = []
    fgclasses = []
    fgscores = []
    flocalisations = []
    fglocalisations = []
    for i in range(len(logits)):
        flogits.append(tf.reshape(logits[i], [-1, num_classes]))
        fgclasses.append(tf.reshape(gclasses[i], [-1]))
        fgscores.append(tf.reshape(gscores[i], [-1]))
        flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
        fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
    # And concat the crap!
    logits = tf.concat(flogits, axis=0)
    gclasses = tf.concat(fgclasses, axis=0)
    gscores = tf.concat(fgscores, axis=0)
    localisations = tf.concat(flocalisations, axis=0)
    glocalisations = tf.concat(fglocalisations, axis=0)
    dtype = logits.dtype

    # Compute positive matching mask...
    pmask = gscores > match_threshold
    fpmask = tf.cast(pmask, dtype)
    n_positives = tf.reduce_sum(fpmask)

    # Hard negative mining...
    no_classes = tf.cast(pmask, tf.int32)
    predictions = slim.softmax(logits)
    nmask = tf.logical_and(tf.logical_not(pmask),
                           gscores > -0.5)
    fnmask = tf.cast(nmask, dtype)
    nvalues = tf.where(nmask,
                       predictions[:, 0],  # voc2012共有目标20类（正样本），加背景（负样本）共21类[0,20]。第0类是背景（负样本）得分
                       1. - fnmask)
    nvalues_flat = tf.reshape(nvalues, [-1])
    # Number of negative entries to select.
    max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
    n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
    n_neg = tf.minimum(n_neg, max_neg_entries)

    val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  # 找背景（负样本）得分最低的n_neg个，所谓的hard negative，最难区分的负样本
    max_hard_pred = -val[-1]
    # Final negative mask.
    nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
    fnmask = tf.cast(nmask, dtype)

    # Add cross-entropy loss.
    # 两个交叉熵函数在label shape上的区别：
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
    # logits[b,w,h,num_classes], gclasses[b,w,h], 真值是0~num_classes之间的一个具体数，不用one-hot
    # loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
    # logits[b,w,h,num_classes], gclasses[b,w,h,num_classes], 真值是num_classes个one-hot的数
    with tf.name_scope('ssd_cls_loss'):
        loss_pos = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
        loss_pos = tf.div(tf.reduce_sum(loss_pos * fpmask), batch_size, name='value')
        loss_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
        loss_neg = tf.div(tf.reduce_sum(loss_neg * fnmask), batch_size, name='value')
        loss_cls = loss_pos + loss_neg

    # Add localization loss: smooth L1, L2, ...
    with tf.name_scope('ssd_loc_loss'):
        # Weights Tensor: positive mask + random negative.
        weights = tf.expand_dims(alpha * fpmask, axis=-1)
        loss = custom_layers.abs_smooth(localisations - glocalisations)
        loss_loc = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')

    return loss_cls, loss_loc