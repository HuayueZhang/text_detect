#!/usr/bin/env bash
export PYTHONPATH=~/pixel_link/code/pylib/src:$PYTHONPATH
set -x
set -e

export CUDA_VISIBLE_DEVICES=$1

python test_pixel_link_on_any_image.py \
            --checkpoint_path=/home/zhy/pixel_link/models/model.ckpt-249419 \
            --dataset_dir=/home/zhy/pixel_link/dataset/mtwi_2018/mtwi_2018_task3_test/icpr_mtwi_task3/image_test \
            --eval_image_width=1280\
            --eval_image_height=768\
            --pixel_conf_threshold=0.5\
            --link_conf_threshold=0.5\
            --gpu_memory_fraction=-1
