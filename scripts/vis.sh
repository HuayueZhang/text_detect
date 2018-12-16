#!/usr/bin/env bash
# !/bin/bash
export PYTHONPATH=~/pixel_link/code/pylib/src:$PYTHONPATH

python visualize_detection_result.py \
	--image=/home/zhy/pixel_link/dataset/mtwi_2018/data_train_eval_split/split_eval/image_eval \
	--det=~/pixel_link/test/mtwi_2018/model.ckpt-249419-eval/txt \
	--output=~/pixel_link/test/mtwi_2018/model.ckpt-249419-eval/visual_result

# 	--image=/home/zhy/pixel_link/dataset/mtwi_2018/mtwi_2018_task3_test/icpr_mtwi_task3/image_test \
# 训练图片与真值可视化
#python visualize_detection_result.py \
#	--image=/home/zhy/pixel_link/dataset/mtwi_2018/mtwi_2018_train/image_train \
#	--det=~/pixel_link/dataset/mtwi_2018/mtwi_2018_train/txt_train \
#	--output=~/pixel_link/dataset/mtwi_2018/mtwi_2018_train/visual_gt
