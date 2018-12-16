# -*- coding: utf-8 -*

from shapely.geometry import Polygon

import os
import math
import itertools


def _str_list_to_polygon(nums):

    if len(nums) < 8:
        raise IndexError()

    points = []
    for idx in range(4):
        points.append((float(nums[idx*2]), float(nums[idx*2+1])))

    def _get_valid_polygon(_pts):

        poly_rect = Polygon(_pts)
        mx, my, xx, xy = poly_rect.bounds

        pad = 100
        mx, my, xx, xy = mx-pad, my-pad, xx+pad, xy+pad
        try:
            poly_rect.intersection(Polygon([(mx, my), (mx, xy), (xx, xy), (xx, my)]))
        except Exception:
            return False, None

        return True, poly_rect

    for pts in itertools.permutations(points, 4):
        do_get_valid, valid_polygon = _get_valid_polygon(pts)
        if do_get_valid:
            return valid_polygon

    return None


def _penal(k):
    return 1 / (1 + math.log(k))


def evaluate_each_picture(drs, gts, to_ignore_gt, t_many, t_one, t_ignore):

    # remove detection results which is to be ignored
    to_ignore_dr = [False] * len(drs)
    for gt_idx, gt in enumerate(gts):
        if not to_ignore_gt[gt_idx]:
            continue

        for dr_idx, dr in enumerate(drs):
            if gt.intersection(dr).area / dr.area > t_ignore:
                to_ignore_dr[dr_idx] = True

    drs = [dr for idx, dr in enumerate(drs) if not to_ignore_dr[idx]]
    gts = [gt for idx, gt in enumerate(gts) if not to_ignore_gt[idx]]

    precision, recall = [0.0] * len(drs), [0.0] * len(gts)

    for gt_idx, gt in enumerate(gts):
        matching_dr_idxes = []
        overlap_sum = 0.0
        for dr_idx, dr in enumerate(drs):
            overlap = gt.intersection(dr)

            if overlap.area / dr.area > t_many:
                matching_dr_idxes.append(dr_idx)
                overlap_sum += overlap.area / gt.area

        if overlap_sum > t_one:
            recall[gt_idx] = 1.0

            for dr_idx in matching_dr_idxes:
                precision[dr_idx] = _penal(len(matching_dr_idxes))

    for dr_idx, dr in enumerate(drs):
        matched_gt_idxes = []
        overlap_sum = 0.0
        for gt_idx, gt in enumerate(gts):
            overlap = dr.intersection(gt)
            if overlap.area / gt.area > t_many:
                matched_gt_idxes.append(gt_idx)
                overlap_sum += overlap.area / dr.area

        if overlap_sum > t_one:
            precision[dr_idx] = 1.0

            for gt_idx in matched_gt_idxes:
                recall[gt_idx] = _penal(len(matched_gt_idxes))

    return sum(precision), sum(recall), len(drs), len(gts)


def zhy_evaluate(dr_txt_path, gt_txt_path, t_many=0.7, t_one=0.7, t_ignore=0.5):

    precision_sum, recall_sum, detection_result_count, ground_truth_count = 0.0, 0.0, 0, 0
    for txt_file in os.listdir(dr_txt_path):
        if not os.path.exists(os.path.join(gt_txt_path, txt_file)):
            raise FileExistsError("")

        detection_results = [_str_list_to_polygon(dr.split(","))
                             for dr in open(os.path.join(dr_txt_path, txt_file), "r")]

        ground_truth_lines = [gt.split(",") for gt in open(os.path.join(gt_txt_path, txt_file), "r")]
        ground_truths = [_str_list_to_polygon(line) for line in ground_truth_lines]

        to_ignore_ground_truth = [line[-1] == "###\n" for line in ground_truth_lines]

        precision, recall, n_valid_dr, n_valid_gt = \
            evaluate_each_picture(detection_results, ground_truths, to_ignore_ground_truth, t_many, t_one, t_ignore)

        detection_result_count += n_valid_dr
        ground_truth_count += n_valid_gt
        precision_sum += precision
        recall_sum += recall

    final_precision = precision_sum / detection_result_count
    final_recall = recall_sum / ground_truth_count
    final_score = 2*final_precision*final_recall / (final_precision + final_recall)

    return final_precision, final_recall, final_score


if __name__ == "__main__":
    p, r, f = zhy_evaluate('/home/zhy/pixel_link/test/mtwi_2018/model.ckpt-84293-dsst-eval/txt', '/home/zhy/pixel_link/dataset/mtwi_2018/data_train_eval_split/split_eval/txt_eval')
    print(p)
    print(r)
    print(f)

