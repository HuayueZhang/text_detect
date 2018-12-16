# -*- coding: utf-8 -*

import numpy as np
import math
import tensorflow as tf
import util
import cv2
import pixel_link
from nets import pixel_link_symbol1
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

slim = tf.contrib.slim
import config

def show_result(image_data, bboxes):
    def draw_bbox(image_data, line, color):
        # line = list(line)
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # data = line.split(',')
        points = [int(v) for v in line[0:8]]
        points = np.reshape(points, (4, 2))
        cnts = util.img.points_to_contours(points)
        util.img.draw_contours(image_data, cnts, -1, color = color, border_width = 3)

    for line in bboxes:
        draw_bbox(image_data, line, color=util.img.COLOR_GREEN)
    plt.imshow(image_data)
    plt.show()


def joint_bbx(bbx0, bbx1, direction, theta, image_data):
    # direction = 1 竖直方向拼接
    # direction = 2 水平方向拼接
    joined_bbx = np.zeros_like(bbx0)
    if direction == 1:
        if theta < 0:
            joined_bbx[0] = min(bbx0[0], bbx0[2], bbx1[0], bbx1[2])
            joined_bbx[1] = max(bbx0[1], bbx0[3], bbx1[1], bbx1[3])
            joined_bbx[2] = max(bbx0[0], bbx0[2], bbx1[0], bbx1[2])
            joined_bbx[3] = min(bbx0[1], bbx0[3], bbx1[1], bbx1[3])
            joined_bbx[4] = max(bbx0[4], bbx0[6], bbx1[4], bbx1[6])
            joined_bbx[5] = min(bbx0[5], bbx0[7], bbx1[5], bbx1[7])
            joined_bbx[6] = min(bbx0[4], bbx0[6], bbx1[4], bbx1[6])
            joined_bbx[7] = max(bbx0[5], bbx0[7], bbx1[5], bbx1[7])
        else:
            joined_bbx[0] = max(bbx0[0], bbx0[2], bbx1[0], bbx1[2])
            joined_bbx[1] = max(bbx0[1], bbx0[3], bbx1[1], bbx1[3])
            joined_bbx[2] = min(bbx0[0], bbx0[2], bbx1[0], bbx1[2])
            joined_bbx[3] = min(bbx0[1], bbx0[3], bbx1[1], bbx1[3])
            joined_bbx[4] = min(bbx0[4], bbx0[6], bbx1[4], bbx1[6])
            joined_bbx[5] = min(bbx0[5], bbx0[7], bbx1[5], bbx1[7])
            joined_bbx[6] = max(bbx0[4], bbx0[6], bbx1[4], bbx1[6])
            joined_bbx[7] = max(bbx0[5], bbx0[7], bbx1[5], bbx1[7])

    if direction == 2:
        if theta < 0:
            joined_bbx[0] = min(bbx0[0], bbx0[6], bbx1[0], bbx1[6])
            joined_bbx[1] = max(bbx0[1], bbx0[7], bbx1[1], bbx1[7])
            joined_bbx[2] = min(bbx0[2], bbx0[4], bbx1[2], bbx1[4])
            joined_bbx[3] = max(bbx0[3], bbx0[5], bbx1[3], bbx1[5])
            joined_bbx[4] = max(bbx0[2], bbx0[4], bbx1[2], bbx1[4])
            joined_bbx[5] = min(bbx0[3], bbx0[5], bbx1[3], bbx1[5])
            joined_bbx[6] = max(bbx0[0], bbx0[6], bbx1[0], bbx1[6])
            joined_bbx[7] = min(bbx0[1], bbx0[7], bbx1[1], bbx1[7])
        else:
            joined_bbx[0] = min(bbx0[0], bbx0[6], bbx1[0], bbx1[6])
            joined_bbx[1] = min(bbx0[1], bbx0[7], bbx1[1], bbx1[7])
            joined_bbx[2] = min(bbx0[2], bbx0[4], bbx1[2], bbx1[4])
            joined_bbx[3] = min(bbx0[3], bbx0[5], bbx1[3], bbx1[5])
            joined_bbx[4] = max(bbx0[2], bbx0[4], bbx1[2], bbx1[4])
            joined_bbx[5] = max(bbx0[3], bbx0[5], bbx1[3], bbx1[5])
            joined_bbx[6] = max(bbx0[0], bbx0[6], bbx1[0], bbx1[6])
            joined_bbx[7] = max(bbx0[1], bbx0[7], bbx1[1], bbx1[7])

    # image_data1 = image_data.copy()
    # show_result(image_data1, [bbx0, bbx1])

    image_data2 = image_data.copy()
    show_result(image_data2, [joined_bbx])
    return joined_bbx


def modify_bboxes(image_data, bboxes):
    adjoin_th = 15
    adjoin_th_max = 30
    scale_th = 15
    theta_th = 0.2
    ratio_th_low = 0.7
    ratio_th_high = 1.5

    flag = 1
    while(flag):
        flag = 0
        for i, bbx0 in enumerate(bboxes):
            if bbx0==None: continue
            p1 = np.array([bbx0[0], bbx0[1]])
            p2 = np.array([bbx0[2], bbx0[3]])
            p3 = np.array([bbx0[4], bbx0[5]])
            p4 = np.array([bbx0[6], bbx0[7]])
            h = dist(p1, p2)
            w = dist(p1, p4)
            t = (p4[1] - p1[1]) / (p4[0] - p1[0] + 0.0001)
            r = w / h

            # image_data1 = image_data.copy()
            # show_result(image_data1, [bbx0])

            for j, bbx1 in enumerate(bboxes):
                if i == j or bbx1 == None: continue
                if i == 8:
                    image_data1 = image_data.copy()
                    show_result(image_data1, [bbx0, bbx1])

                pp1 = np.array([bbx1[0], bbx1[1]])
                pp2 = np.array([bbx1[2], bbx1[3]])
                pp3 = np.array([bbx1[4], bbx1[5]])
                pp4 = np.array([bbx1[6], bbx1[7]])
                hh = dist(pp1, pp2)
                ww = dist(pp1, pp4)
                tt = (pp4[1] - pp1[1]) / (pp4[0] - pp1[0] + 0.0001)
                if abs(t-tt)<theta_th:
                    # 瘦高型bbx尝试纵向拼接
                    if r<1/ratio_th_low and abs(w-ww)<scale_th and (dist(p1, pp2)<adjoin_th or dist(p2, pp1)<adjoin_th):
                        bbx0 = joint_bbx(bbx0, bbx1, 1, t, image_data)
                    # 矮胖型bbx尝试横向拼接
                    elif r>=ratio_th_high and abs(h-hh)<scale_th and (dist(p1, pp4)<adjoin_th or dist(p4, pp1)<adjoin_th):
                        bbx0 = joint_bbx(bbx0, bbx1, 2, t, image_data)
                    # 方型bbx尝试纵向拼接（允许距离范围更广）
                    elif ratio_th_low<=r<=ratio_th_high and abs(w-ww)<scale_th and \
                            (dist(p1, pp2)<adjoin_th_max or dist(p2, pp1)<adjoin_th_max):
                        bbx0 = joint_bbx(bbx0, bbx1, 1, t, image_data)
                    # 方型bbx尝试横向拼接（允许距离范围更广）
                    elif ratio_th_low<=r<=ratio_th_high and abs(h-hh)<scale_th and  \
                            (dist(p1, pp4)<adjoin_th_max or dist(p4, pp1)<adjoin_th_max):
                        bbx0 = joint_bbx(bbx0, bbx1, 2, t, image_data)
                    else:
                        continue

                    bboxes[j] = None
                    p1 = np.array([bbx0[0], bbx0[1]])
                    p2 = np.array([bbx0[2], bbx0[3]])
                    p3 = np.array([bbx0[4], bbx0[5]])
                    p4 = np.array([bbx0[6], bbx0[7]])
                    h = dist(p1, p2)
                    w = dist(p1, p4)
                    t = (p4[1] - p1[1]) / (p4[0] - p1[0] + 0.0001)
                    r = w / h
                    flag = 1

            bboxes[i] = bbx0
        while None in bboxes:
            bboxes.remove(None)

    modified_bboxes = bboxes
    return modified_bboxes

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def fill_in_mask(mask, xys):
    util.img.draw_contours(mask, xys, -1, 1, border_width=-1)
    return mask

def do_choose(bbox_mask, bbox_mask1, bbox_mask2, d, d1, d2, th1, th2):
    # plt.figure(1)
    # plt.imshow(bbox_mask1+bbox_mask2)
    # plt.figure(2)
    # plt.imshow(bbox_mask)
    # plt.show()

    cross1 = (bbox_mask + bbox_mask1) == 2
    cross2 = (bbox_mask + bbox_mask2) == 2
    s1 = bbox_mask1.sum()
    s2 = bbox_mask2.sum()
    s = bbox_mask.sum()
    if cross1.sum() > 0.9*s1 and cross2.sum > 0.9*s2:
        if s1+s2 > th1*s and d-d1-d2 < th2:
            return 'Joined'
    return 'Origin'


# @util.dec.print_calling_in_short
# @util.dec.timeit
def modify_mask_to_bboxes(mask, image_shape=None, min_area=None,
                   min_height=None, min_aspect_ratio=None):
    import config
    feed_shape = config.train_image_shape
    scale_th = 15

    if image_shape is None:
        image_shape = feed_shape

    image_h, image_w = image_shape[0:2]

    if min_area is None:
        min_area = config.min_area

    if min_height is None:
        min_height = config.min_height

    max_bbox_idx = mask.max()

    mask = util.img.resize(img=mask, size=(image_w, image_h),
                           interpolation=cv2.INTER_NEAREST)

    masks = []
    cnts = []
    rects = []
    for bbox_idx in xrange(1, max_bbox_idx+1):
        mask_temp = mask == bbox_idx

        cnt = util.img.find_contours(mask_temp)  # 轮廓点

        if len(cnt) == 0: continue
        rect, rect_area = pixel_link.min_area_rect(cnt[0])  # 中点坐标+宽高
        w, h = rect[2:-1]
        if min(w, h) < min_height: continue
        if rect_area < min_area: continue

        xys = pixel_link.rect_to_xys(rect, image_shape)  # 4个顶点坐标
        mask_temp = np.zeros(image_shape[0: -1])
        mask_temp = fill_in_mask(mask_temp, [xys.reshape((-1, 1, 2))])

        # plt.figure(1)
        # plt.imshow(mask_temp)
        # plt.show()

        masks.append(mask_temp)
        # cnts.append(cnt[0])
        cnts.append(xys.reshape((-1, 1, 2)))
        rects.append([w, h])


    flag = 1
    while(flag):
        flag = 0
        for idx1, bbox_mask1 in enumerate(masks):
            # idx1 = 5
            # bbox_mask1 = masks[idx1]
            if bbox_mask1 == None: continue
            cnt1 = cnts[idx1]
            w1, h1 = rects[idx1]

            for idx2, bbox_mask2 in enumerate(masks):
                # idx2 = 8
                # bbox_mask2 = masks[idx2]
                if idx1 == idx2 or bbox_mask2 == None: continue
                cnt2 = cnts[idx2]
                w2, h2 = rects[idx2]
                #
                # plt.figure(1)
                # plt.imshow(bbox_mask1)
                # plt.figure(2)
                # plt.imshow(bbox_mask2)
                # plt.show()

                if 0.7<w1/h1<1.4 and (abs(w1-w2)<scale_th or abs(w1-h2)<scale_th):
                    keep = None
                    if 0.7<w2/h2<1.4:
                        th2 = 1e4
                        th1 = 0.7  # 无谓方向，放大阈值
                    else:
                        th2 = 20.0
                        th1 = 0.9  # 无谓方向，放大阈值
                elif w1/h1<=0.7 and (abs(w1-w2)<scale_th or abs(w1-h2)<scale_th):
                    keep = 'w1'   # 拼接保持w1不变（小者）
                    th1 = 0.9
                    th2 = 15.0
                elif w1/h1>=1.4 and (abs(h1-w2)<scale_th or abs(h1-h2)<scale_th):
                    keep = 'h1'   # 拼接保持h1不变（小者）
                    th1 = 0.9
                    th2 = 15.0
                else:
                    continue

                # plt.figure(1)
                # plt.imshow(bbox_mask1)
                # plt.figure(2)
                # plt.imshow(bbox_mask2)
                # plt.show()

                cnt = np.vstack((cnt1, cnt2))
                rect, rect_area = pixel_link.min_area_rect(cnt)
                w, h = rect[2:-1]
                if min(w, h) < min_height: continue
                if rect_area < min_area: continue
                xys = pixel_link.rect_to_xys(rect, image_shape)  # 4个顶点坐标

                bbox_mask = np.zeros_like(bbox_mask1, dtype = np.int32)
                bbox_mask = fill_in_mask(bbox_mask, [xys.reshape((-1,1,2))])
                choice = do_choose(bbox_mask, bbox_mask1, bbox_mask2, max(w, h), max(w1, h1), max(w2, h2), th1, th2)

                if choice == 'Joined':

                    # plt.figure(1)
                    # plt.imshow(bbox_mask1 + bbox_mask2)
                    # plt.figure(2)
                    # plt.imshow(bbox_mask)
                    # plt.show()

                    cnt_temp = util.img.find_contours(bbox_mask)
                    rect_temp, rect_area_temp = pixel_link.min_area_rect(cnt_temp[0])  # 中点坐标+宽高
                    w_temp, h_temp = rect_temp[2:-1]
                    if not keep == None:   # keep 'w1' or 'h1'
                        if not abs(min(w1, h1)-min(w_temp, h_temp)) < min(scale_th, min(w1, h1)/3):
                            continue

                    # plt.figure(1)
                    # plt.imshow(bbox_mask1 + bbox_mask2)
                    # plt.figure(2)
                    # plt.imshow(bbox_mask)
                    # plt.show()

                    masks[idx2] = None
                    cnts[idx2] = None
                    rects[idx2] = None
                    bbox_mask1 = bbox_mask

                    xys1 = pixel_link.rect_to_xys(rect_temp, image_shape)

                    cnt1 = xys1.reshape((-1, 1, 2))
                    w1 = w_temp
                    h1 = h_temp
                    flag = 1

            masks[idx1] = bbox_mask1
            cnts[idx1] = cnt1
            rects[idx1] = [w1, h1]
            while None in masks:
                masks.remove(None)
            while None in cnts:
                cnts.remove(None)
            while None in rects:
                rects.remove(None)

    # for idx, bbox_mask in enumerate(masks):
    #     if bbox_mask == None: continue
    #     cnts = util.img.find_contours(bbox_mask)  # 轮廓点
    #     if len(cnts) == 0: continue
    #     cnt = cnts[0]
    #     rect, rect_area = pixel_link.min_area_rect(cnt)  # 中点坐标+宽高
    #     w, h = rect[2:-1]
    #     if min(w, h) < min_height: continue
    #     if rect_area < min_area: continue
    #     xys = pixel_link.rect_to_xys(rect, image_shape)  # 4个顶点坐标
    #     bboxes.append(xys)

    bboxes = []
    for line in cnts:
        line1 = line.reshape((8,))
        bboxes.append(line1)

    return bboxes