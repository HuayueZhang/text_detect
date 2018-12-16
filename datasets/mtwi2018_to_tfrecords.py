#encoding=utf-8
import os
import numpy as np
import util
import tensorflow as tf
import config
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example
import shutil

def to_valid(v):
    if v>1:
        return 1
    if v<0:
        return 0
    return v

def cvt_to_tfrecords(image_names, output_path, data_path, gt_path, begin, end):

    with tf.python_io.TFRecordWriter(output_path) as tfrecord_writer:
        ii = 0
        for idx, image_name in enumerate(image_names[begin:end]):
            oriented_bboxes = []
            bboxes = []
            bboxesn = []
            labels = []
            labels_text = []
            path = util.io.join_path(data_path, image_name)
            image_data = tf.gfile.FastGFile(path, 'r').read()
            try:
                image = util.img.imread(path, rgb = True)
                ii = ii+1
            except:
                continue
            print "\tconverting image: %d/%d %s" % (ii, len(image_names), image_name)
            shape = image.shape
            h, w = shape[0:2]
            h *= 1.0
            w *= 1.0
            image_name = image_name[0:-4]
            gt_name = image_name + '.txt'
            gt_filepath = util.io.join_path(gt_path, gt_name)
            lines = util.io.read_lines(gt_filepath)

            for line in lines:
                line = util.str.remove_all(line, '\xef\xbb\xbf')
                gt = util.str.split(line, ',')
                oriented_box = [int(float(gt[i])) for i in range(8)]
                oriented_box = np.asarray(oriented_box) / ([w, h] * 4)
                oriented_bboxes.append(oriented_box)

                xs = oriented_box.reshape(4, 2)[:, 0]
                ys = oriented_box.reshape(4, 2)[:, 1]

                xmin = to_valid(xs.min())
                xmax = to_valid(xs.max())
                ymin = to_valid(ys.min())
                ymax = to_valid(ys.max())

                bboxes.append([xmin, ymin, xmax, ymax])

                labels_text.append(gt[-1])
                ignored = util.str.contains(gt[-1], '###')
                if ignored:
                    labels.append(config.ignore_label)
                else:
                    labels.append(config.text_label)
            example = convert_to_example(image_data, image_name, labels, labels_text, bboxes, oriented_bboxes, shape)
            tfrecord_writer.write(example.SerializeToString())
        # print 'done!'

def split_train_data(image_names, output_path, data_path, gt_path, begin, end):
    eval_image_path = util.io.join_path(output_path, 'image_eval')
    eval_gt_path = util.io.join_path(output_path, 'txt_eval')

    for idx, image_name in enumerate(image_names[begin:end]):
        print idx
        img_filepath = util.io.join_path(data_path, image_name)
        try:
            image = util.img.imread(img_filepath, rgb=True)
        except:
            continue

        image_name = image_name[0:-4]
        gt_name = image_name + '.txt'
        gt_filepath = util.io.join_path(gt_path, gt_name)

        shutil.copy(img_filepath, eval_image_path)
        shutil.copy(gt_filepath, eval_gt_path)

if __name__ == "__main__":
    root_dir = util.io.get_absolute_path('~/pixel_link/')
    data_path = util.io.join_path(root_dir, 'dataset/mtwi_2018')
    train_path = util.io.join_path(data_path, 'mtwi_2018_train/image_train')
    train_gt_path = util.io.join_path(data_path, 'mtwi_2018_train/txt_train')

    image_names = util.io.ls(train_path, '.jpg')
    print "%d images found in %s" % (len(image_names), data_path)

    # cvt_to_tfrecords(image_names=image_names,
    #                  output_path = util.io.join_path(data_path, 'data_train_eval_split/split_train.tfrecord'),
    #                  data_path = train_path,
    #                  gt_path = train_gt_path,
    #                  begin=0,
    #                  end=7100)
    #
    # cvt_to_tfrecords(image_names=image_names,
    #                  output_path = util.io.join_path(data_path, 'data_train_eval_split/split_eval.tfrecord'),
    #                  data_path = train_path,
    #                  gt_path = train_gt_path,
    #                  begin=7100,
    #                  end=len(image_names))
    split_train_data(image_names=image_names,
                     output_path='/home/zhy/pixel_link/dataset/mtwi_2018/data_train_eval_split/split_eval',
                     data_path=train_path,
                     gt_path=train_gt_path,
                     begin=7100,
                     end=len(image_names))

    # test_path = os.path.join(data_path, 'mtwi_2018_task3_test/icpr_mtwi_task3/image_test')
    # cvt_to_tfrecords(output_path = os.path.join(data_path, 'mtwi_2018_task3_test.tfrecords'),
    #                  data_path = test_path, gt_path = None)
