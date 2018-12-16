# -*- coding: utf-8 -*

import numpy as np
import tensorflow as tf

import config
from nets import ssd_vgg_300

num_classes = 5

# get the ssd network and its anchors
ssd_class = ssd_vgg_300.SSDNet
ssd_net = ssd_class()
ssd_shape = config.img_shape
ssd_anchors = ssd_net.anchors(ssd_shape)

def ssd_loss(logits, localisations,
             gclasses, glocalisations, gscores):
    return ssd_net.losses(logits, localisations,
                          gclasses, glocalisations, gscores)


def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

# encode groundtruth labels and bboxes
def tf_encode_gt(labels, ys, xs):
    point_labels, point_bboxes = tf.py_func(
        encode_gt, [xs, ys, labels],
        [tf.int64, tf.float32])

    point_bboxes.set_shape([None, 4])
    point_labels.set_shape([None,])

    return ssd_net.bboxes_encode(point_labels, point_bboxes, ssd_anchors)

def encode_gt(xs, ys, labels):
    """
    Args:
        xs, ys: both in shape of (N, 4),
            and N is the number of bboxes,
            their values are normalized to [0,1]  ???????????????????????????????????
        labels: shape = (N,), only two values are allowed:
                                                        -1: ignored
                                                        1: text
    Return:
        point_labels
        point_bboxes consisting of [ymin, xmin, ymax, xmax]s
    """
    point_labels = []
    point_bboxes = []
    for bbox_idx, (bbox_ys, bbox_xs) in enumerate(zip(ys, xs)):
        bbox_points = zip(bbox_ys, bbox_xs)
        ss = min(dist(np.array(bbox_points[0]), np.array(bbox_points[1])),
                 dist(np.array(bbox_points[0]), np.array(bbox_points[2])))
        idx_xs = np.argsort(bbox_xs)
        if bbox_ys[idx_xs[0]] > bbox_ys[idx_xs[1]]:
            p0 = bbox_points[idx_xs[0]]
            p3 = bbox_points[idx_xs[1]]
            p1 = bbox_points[idx_xs[2]]
            p2 = bbox_points[idx_xs[3]]

        else:
            p3 = bbox_points[idx_xs[0]]
            p0 = bbox_points[idx_xs[1]]
            p2 = bbox_points[idx_xs[2]]
            p1 = bbox_points[idx_xs[3]]

        point_bboxes.append(list([(p0 - ss / 2) + (p0 + ss / 2)]))
        point_bboxes.append(list([(p1 - ss / 2) + (p1 + ss / 2)]))
        point_bboxes.append(list([(p2 - ss / 2) + (p2 + ss / 2)]))
        point_bboxes.append(list([(p3 - ss / 2) + (p3 + ss / 2)]))

        point_labels.append([1])
        point_labels.append([2])
        point_labels.append([3])
        point_labels.append([4])


    return point_labels, point_bboxes













