# -*- coding: utf-8 -*
""" Algorithm evaluators for Rigor """

from __future__ import print_function
from collections import defaultdict
import sys

class ObjectAreaEvaluator(object):
    """
    Compares ground truth to detections using Wolf and Jolion's algorithm.

    :param scatter_punishment: :math:`f_{sc}(k)` "a parameter function of the evaluation scheme which controls the amount of punishment which is inflicted in case of scattering, i.e. splits or merges"
    :type scatter_punishment: lambda(x): -> float
    :param float precision_threshold: :math:`t_{p}` in [1]_
    :param float recall_threshold: :math:`t_{r}` in [1]_

    .. seealso::

        Object count/Area Graphs for the Evaluation of Object Detection and Segmentation Algorithms [1]_

        .. [1] http://liris.cnrs.fr/Documents/Liris-2216.pdf
    """

    import numpy as np
    from shapely.geometry import Polygon

    def __init__(self, scatter_punishment=lambda(k): 1.0, precision_threshold=0.4, recall_threshold=0.8):
        self.scatter_punishment = scatter_punishment
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold

    @staticmethod
    def non_zero_polygon(polygon, suppress_warning=False):
        """
        Checks that a polygon has a nonzero area. If the area is zero, it will be
        dilated by a small amount so that overlap and such can be measured.

        :param polygon: The polygon to test
        :type polygon: :py:class:`~shapely.Polygon`
        :param bool suppress_warning: If :py:const:`False`, a warning will be printed if the area is dilated; if :py:const:`True`, no warning will be printed.
        :return: the original :py:class:`~shapely.Polygon`, possibly dilated
        """
        if polygon.area > 0:
            return polygon

        if not suppress_warning:
            print("Warning: polygon has zero area; dilating", file=sys.stderr)
        return polygon.buffer(0.05, 1).convex_hull

    @classmethod
    def prune_and_polygon(cls, ground_truths, detections):
        """
        Given either :py:class:`~shapely.Polygon` instances or plain-Python sequences of
        vertices, returns a tuple of ground truth and detection :py:class:`~shapely.Polygon`
        instances, excluding ground truth polygons that have zero length
        """
        if not hasattr(ground_truths[0], 'intersection'):
            ground_truths = [cls.Polygon(value) for value in ground_truths]
        if not hasattr(detections[0], 'intersection'):
            detections = [cls.Polygon(value) for value in detections]
        ground_truths = [value for value in ground_truths if value.length > 0.]
        return (ground_truths, detections)

    @classmethod
    def build_matrices(cls, ground_truths, detections):
        """
        Builds a set of matrices containing measurements of overlap between ground
        truth and detections.

        :param ground_truths: Sequence of ground truth polygons
        :param detections: Sequence of detected polygons
        :return: Tuple of :py:class:`numpy.array` arrays: (ground truth matches, detection matches)
        """
        ground_truth_count = len(ground_truths)
        detection_count = len(detections)

        recall_matrix = cls.np.empty((ground_truth_count, detection_count), dtype=float)
        precision_matrix = cls.np.empty((ground_truth_count, detection_count), dtype=float)

        for gt_index in range(ground_truth_count):
            ground_truth = ObjectAreaEvaluator.non_zero_polygon(ground_truths[gt_index])
            for det_index in range(detection_count):
                detection = ObjectAreaEvaluator.non_zero_polygon(detections[det_index])
                overlap_polygon = ground_truth.intersection(detection)

                precision_area = overlap_polygon.area / detection.area
                precision_matrix[gt_index, det_index] = precision_area
                recall_area = overlap_polygon.area / ground_truth.area
                recall_matrix[gt_index, det_index] = recall_area
        return (precision_matrix, recall_matrix)

    def match_detections(self, ground_truths, detections):
        """ Compares ground_truths to detections """
        if not ground_truths or not detections:
            return (0., 0., (0., len(detections)), (0., len(ground_truths)))

        ground_truths, detections = ObjectAreaEvaluator.prune_and_polygon(ground_truths, detections)
        ground_truth_count = len(ground_truths)
        detection_count = len(detections)
        if ground_truth_count == 0 or detection_count == 0:
            return (0., 0., (0., float(detection_count)), (0., float(ground_truth_count)))

        precision_matrix, recall_matrix = self.build_matrices(ground_truths, detections)
        ground_truth_count = precision_matrix.shape[0]
        detection_count = precision_matrix.shape[1]
        ground_truth_sets_precision = defaultdict(set) # number of ground truth items that match a particular detection in the precision matrix
        detection_sets_precision = defaultdict(set) # number of detection items that match a particular ground truth in the precision matrix
        ground_truth_sets_recall = defaultdict(set) # number of ground truth items that match a particular detection in the recall matrix
        detection_sets_recall = defaultdict(set) # number of detection items that match a particular ground truth in the recall matrix

        for gt_index in range(ground_truth_count):
            for det_index in range(detection_count):
                if precision_matrix[gt_index, det_index] >= self.precision_threshold:
                    ground_truth_sets_precision[det_index].add(gt_index)
                    detection_sets_precision[gt_index].add(det_index)
                if recall_matrix[gt_index, det_index] >= self.recall_threshold:
                    ground_truth_sets_recall[det_index].add(gt_index)
                    detection_sets_recall[gt_index].add(det_index)

        match_ground_truth =  [list() for _ in range(ground_truth_count)]
        match_detection = [list() for _ in range(detection_count)]

        for gt_index in detection_sets_precision:
            matching_detections_precision = detection_sets_precision[gt_index]
            if len(matching_detections_precision) == 1:
                (detection_precision, ) = matching_detections_precision
                if len(ground_truth_sets_precision[detection_precision]) == 1:
                    match_ground_truth[gt_index].append(detection_precision)
            else:
                # one-to-many (one ground truth to many detections)
                gt_sum = 0.
                for detection_precision in matching_detections_precision:
                    gt_sum += recall_matrix[gt_index, detection_precision]
                if gt_sum >= self.recall_threshold:
                    for detection_precision in matching_detections_precision:
                        match_ground_truth[gt_index].append(detection_precision)
                        match_detection[detection_precision].append(gt_index)
        for det_index in ground_truth_sets_recall:
            matching_ground_truths_recall = ground_truth_sets_recall[det_index]
            if len(matching_ground_truths_recall) == 1:
                (ground_truth_recall, ) = matching_ground_truths_recall
                if len(detection_sets_recall[ground_truth_recall]) == 1:
                    match_detection[det_index].append(ground_truth_recall)
            else:
                # many-to-one (many ground truths covered by one detection)
                det_sum = 0
                for ground_truth_recall in matching_ground_truths_recall:
                    det_sum += precision_matrix[ground_truth_recall, det_index]
                if det_sum >= self.precision_threshold:
                    for ground_truth_recall in matching_ground_truths_recall:
                        det_sum += precision_matrix[ground_truth_recall, det_index]
                        match_detection[det_index].append(ground_truth_recall)
                        match_ground_truth[ground_truth_recall].append(det_index)
        return match_ground_truth, match_detection

    def evaluate(self, ground_truths, detections):
        r"""
        Given lists of polylines for each parameter (ground_truths, detections),
        this will check the overlap and return a (precision, recall, (:math:`\sum Match_D`,
        :math:`|D|`), (:math:`\sum Match_G`, :math:`|G|`)) tuple for the overall image.

        ground_truths and detections should both be sequences of (x,y) point tuples.
        """
        # (0.5714285714285714, 0.8, (4.0, 7.0), (4.0, 5.0))

        if not ground_truths or not detections:
            return (0.,0.,(0.,len(detections)), (0., len(ground_truths)))

        ground_truths, detections = ObjectAreaEvaluator.prune_and_polygon(ground_truths, detections)
        ground_truth_count = len(ground_truths)
        detection_count = len(detections)
        if ground_truth_count == 0 or detection_count == 0:
            return (0., 0., (0., float(detection_count)), (0., float(ground_truth_count)))

        precision_matrix, recall_matrix = self.build_matrices(ground_truths, detections)
        return self.evaluate_matrices(precision_matrix, recall_matrix)

    def evaluate_matrices(self, precision_matrix, recall_matrix):
        """
        Given a precision and recall matrix (2d matrix; rows are ground truth,
        columns are detections) containing overlap between each pair of ground
        truth and detection polygons, this will run the match functions over the
        matrix and return a (precision, recall, (:math:`\sum Match_D`, :math:`|D|`), (:math:`\sum Match_G`,
        :math:`|G|`)) tuple for the overall image.
        """
        ground_truth_count = precision_matrix.shape[0]
        detection_count = precision_matrix.shape[1]
        ground_truth_sets_precision = defaultdict(set) # number of ground truth items that match a particular detection in the precision matrix
        detection_sets_precision = defaultdict(set) # number of detection items that match a particular ground truth in the precision matrix
        ground_truth_sets_recall = defaultdict(set) # number of ground truth items that match a particular detection in the recall matrix
        detection_sets_recall = defaultdict(set) # number of detection items that match a particular ground truth in the recall matrix

        for gt_index in range(ground_truth_count):
            for det_index in range(detection_count):
                if precision_matrix[gt_index, det_index] >= self.precision_threshold:
                    ground_truth_sets_precision[det_index].add(gt_index)
                    detection_sets_precision[gt_index].add(det_index)
                if recall_matrix[gt_index, det_index] >= self.recall_threshold:
                    ground_truth_sets_recall[det_index].add(gt_index)
                    detection_sets_recall[gt_index].add(det_index)

        match_ground_truth = 0. # sum of MatchG
        match_detection = 0. # sum of MatchD

        one_to_one_precision = set()
        for gt_index in detection_sets_precision:
            matching_detections_precision = detection_sets_precision[gt_index]
            if len(matching_detections_precision) == 1:
                (detection_precision, ) = matching_detections_precision
                if len(ground_truth_sets_precision[detection_precision]) == 1:
                    one_to_one_precision.add((gt_index, detection_precision))
            else:
                # one-to-many (one ground truth to many detections)
                gt_sum = 0.
                for detection_precision in matching_detections_precision:
                    gt_sum += recall_matrix[gt_index, detection_precision]
                if gt_sum >= self.recall_threshold:
                    #print("1:N ~ GT {} : DT {}".format(gt_index,matching_detections_precision))
                    match_ground_truth += self.scatter_punishment(matching_detections_precision)
                    match_detection += len(matching_detections_precision) * self.scatter_punishment(matching_detections_precision)

        one_to_one_recall = set()
        for det_index in ground_truth_sets_recall:
            matching_ground_truths_recall = ground_truth_sets_recall[det_index]
            if len(matching_ground_truths_recall) == 1:
                (ground_truth_recall, ) = matching_ground_truths_recall
                if len(detection_sets_recall[ground_truth_recall]) == 1:
                    one_to_one_recall.add((ground_truth_recall, det_index))
            else:
                # many-to-one (many ground truths covered by one detection)
                det_sum = 0
                for ground_truth_recall in matching_ground_truths_recall:
                    det_sum += precision_matrix[ground_truth_recall, det_index]
                if det_sum >= self.precision_threshold:
                    #print("N:1 ~ DT {} : GT {}".format(det_index,matching_ground_truths_recall))
                    match_detection += self.scatter_punishment(matching_ground_truths_recall)
                    match_ground_truth += len(matching_ground_truths_recall) * self.scatter_punishment(matching_ground_truths_recall)

        one_to_one_matches = one_to_one_precision & one_to_one_recall
        match_ground_truth += len(one_to_one_matches)
        match_detection += len(one_to_one_matches)

        recall = match_ground_truth / float(ground_truth_count)
        precision = match_detection / float(detection_count)
        return (precision, recall, (match_detection, float(detection_count)), (match_ground_truth, float(ground_truth_count)))
