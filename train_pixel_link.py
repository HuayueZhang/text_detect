#test code to make sure the ground truth calculation and data batch works well.

import numpy as np
import tensorflow as tf # test
from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
import util
import pixel_link
slim = tf.contrib.slim
import config

from IPython import embed
import pdb

from ssd import train_ssd_network as ssd_op
from ssd import tf_utils as ssd_utils

# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('train_dir', None, 
                           'the path to store checkpoints and eventfiles for summaries')

tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
                          'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')

tf.app.flags.DEFINE_integer('batch_size', None, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer('gpu_idx', 1, 'The index of gpu.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'The number of gpus can be used.')
tf.app.flags.DEFINE_integer('max_number_of_steps', 1000000, 'The maximum number of training steps.')
tf.app.flags.DEFINE_integer('log_every_n_steps', 1, 'log frequency')
tf.app.flags.DEFINE_bool("ignore_missing_vars", False, '')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None, 'checkpoint_exclude_scopes')

# =========================================================================== #
# Optimizer configs.
# =========================================================================== #
tf.app.flags.DEFINE_float('learning_rate', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer')
tf.app.flags.DEFINE_float('weight_decay', 0.0001, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_bool('using_moving_average', True, 'Whether to use ExponentionalMovingAverage')
tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 'The decay rate of ExponentionalMovingAverage')

# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', None, 'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer('train_image_width', 512, 'Train image size')
tf.app.flags.DEFINE_integer('train_image_height', 512, 'Train image size')


FLAGS = tf.app.flags.FLAGS
def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.train_image_height, FLAGS.train_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    util.init_logger(
        log_file = 'log_train_pixel_link_%d_%d.log' % image_shape,
                    log_path = FLAGS.train_dir, stdout = False, mode = 'a')
    
    config.load_config(FLAGS.train_dir)
    config.init_config(image_shape, 
                       batch_size = FLAGS.batch_size, 
                       weight_decay = FLAGS.weight_decay, 
                       num_gpus = FLAGS.num_gpus,
                       gpu_idx=FLAGS.gpu_idx
                   )

    batch_size = config.batch_size
    batch_size_per_gpu = config.batch_size_per_gpu
        
    tf.summary.scalar('batch_size', batch_size)
    tf.summary.scalar('batch_size_per_gpu', batch_size_per_gpu)

    # util.proc.set_proc_name('train_pixel_link_on'+ '_' + FLAGS.dataset_name)
    util.proc.set_proc_name('python_train')
    
    dataset = dataset_factory.get_dataset(FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)
    config.print_config(FLAGS, dataset)
    return dataset


def create_dataset_batch_queue2(dataset):
    from preprocessing import ssd_vgg_preprocessing

    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * config.batch_size,
                common_queue_min=700 * config.batch_size,
                shuffle=True)
        # Get for SSD network: an image, labels, bboxes.
        [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
            'image',
            'object/label',
            'object/bbox',
            'object/oriented_bbox/x1',
            'object/oriented_bbox/x2',
            'object/oriented_bbox/x3',
            'object/oriented_bbox/x4',
            'object/oriented_bbox/y1',
            'object/oriented_bbox/y2',
            'object/oriented_bbox/y3',
            'object/oriented_bbox/y4'
            ])
        gxs = tf.transpose(tf.stack([x1, x2, x3, x4])) #shape = (N, 4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        image = tf.identity(image, 'input_image')
        
        # Pre-processing image, labels and bboxes.
        image, glabel, gbboxes, gxs, gys = \
                ssd_vgg_preprocessing.preprocess_image(
                       image, glabel, gbboxes, gxs, gys, 
                       out_shape   = config.train_image_shape,
                       data_format = config.data_format, 
                       use_rotation= config.use_rotation,
                       is_training = True)
        image = tf.identity(image, 'processed_image')
        
        # calculate ground truth for an image
        pixel_cls_label, pixel_cls_weight, \
        pixel_link_label, pixel_link_weight = \
            pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)

        # calculate ground truth for default boxes of ssd
        gclasses, glocalisations, gscores = ssd_op.tf_encode_gt(glabel, gys, gxs)

        # batch them
        batch_shape = [1] + [1, 1, 1, 1] + [len(config.feat_layers)] * 3
        to_batch = ssd_utils.reshape_list([image,
                    pixel_cls_label, pixel_cls_weight, pixel_link_label, pixel_link_weight,
                    gclasses, glocalisations, gscores])
        with tf.name_scope(FLAGS.dataset_name + '_batch'):
            r = tf.train.batch(to_batch,
                               batch_size = config.batch_size_per_gpu,
                               num_threads= FLAGS.num_preprocessing_threads,
                               capacity   = 500)

        b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight,\
        b_gclasses, b_glocalisations, b_scores = ssd_utils.reshape_list(r, batch_shape)

        to_queue = ssd_utils.reshape_list(
            [b_image, b_pixel_cls_label, b_pixel_cls_weight, b_pixel_link_label, b_pixel_link_weight,
             b_gclasses, b_glocalisations, b_scores])
        with tf.name_scope(FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(to_queue, capacity = 50)

    return batch_queue


def create_dataset_batch_queue(dataset):
    from preprocessing import ssd_vgg_preprocessing

    with tf.device('/cpu:0'):
        with tf.name_scope(FLAGS.dataset_name + '_data_provider'):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                num_readers=FLAGS.num_readers,
                common_queue_capacity=1000 * config.batch_size,
                common_queue_min=700 * config.batch_size,
                shuffle=True)
        # Get for SSD network: an image, labels, bboxes.
        [image, glabel, gbboxes, x1, x2, x3, x4, y1, y2, y3, y4] = provider.get([
            'image',
            'object/label',
            'object/bbox',
            'object/oriented_bbox/x1',
            'object/oriented_bbox/x2',
            'object/oriented_bbox/x3',
            'object/oriented_bbox/x4',
            'object/oriented_bbox/y1',
            'object/oriented_bbox/y2',
            'object/oriented_bbox/y3',
            'object/oriented_bbox/y4'
        ])
        gxs = tf.transpose(tf.stack([x1, x2, x3, x4]))  # shape = (N, 4)
        gys = tf.transpose(tf.stack([y1, y2, y3, y4]))
        image = tf.identity(image, 'input_image')

        # Pre-processing image, labels and bboxes.
        image, glabel, gbboxes, gxs, gys = \
            ssd_vgg_preprocessing.preprocess_image(
                image, glabel, gbboxes, gxs, gys,
                out_shape=config.train_image_shape,
                data_format=config.data_format,
                use_rotation=config.use_rotation,
                is_training=True)
        image = tf.identity(image, 'processed_image')

        # calculate ground truth for an image
        pixel_cls_label, pixel_cls_weight, \
        pixel_link_label, pixel_link_weight = \
            pixel_link.tf_cal_gt_for_single_image(gxs, gys, glabel)

        # batch them
        with tf.name_scope(FLAGS.dataset_name + '_batch'):
            b_image, b_pixel_cls_label, b_pixel_cls_weight, \
            b_pixel_link_label, b_pixel_link_weight = \
                tf.train.batch(
                    [image, pixel_cls_label, pixel_cls_weight,
                        pixel_link_label, pixel_link_weight],
                    batch_size = config.batch_size_per_gpu,
                    num_threads= FLAGS.num_preprocessing_threads,
                    capacity = 500)
        with tf.name_scope(FLAGS.dataset_name + '_prefetch_queue'):
            batch_queue = slim.prefetch_queue.prefetch_queue(
                [b_image, b_pixel_cls_label, b_pixel_cls_weight,
                    b_pixel_link_label, b_pixel_link_weight],
                capacity = 50)

    return batch_queue


def sum_gradients(clone_grads):                        
    averaged_grads = []
    for grad_and_vars in zip(*clone_grads):
        grads = []
        var = grad_and_vars[0][1]

        for g, v in grad_and_vars:
            assert v == var
            grads.append(g)
        try:
            grad = tf.add_n(grads, name = v.op.name + '_summed_gradients')
        except:
            grad = tf.zeros_like(var)
        
        averaged_grads.append((grad, v))

    return averaged_grads


def create_clones2(batch_queue):
    if config.model_type == 'vgg16':
        from nets import pixel_link_symbol1 as pixel_link_symbol
    elif config.model_type == 'vgg16_dssd':
        from nets import pixel_link_symbol1 as pixel_link_symbol
    else:  # 'vgg16_dssd_ssd'
        from nets import pixel_link_symbol2 as pixel_link_symbol

    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate, 
                               momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0 # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0 # only summary on the first clone
        reuse = clone_idx > 0
        with tf.variable_scope('vgg_dssd', reuse = reuse):
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    r = batch_queue.dequeue()
                    batch_shape = [1] + [1, 1, 1, 1] + [len(config.feat_layers)] * 3
                    b_image, b_pixel_cls_label, b_pixel_cls_weight, \
                        b_pixel_link_label, b_pixel_link_weight, \
                        b_gclasses, b_glocalisations, b_scores = ssd_utils.reshape_list(r, batch_shape)

                    # build model and loss
                    net = pixel_link_symbol.PixelLinkNet(b_image, is_training = True)
                    variables_to_train = tf.trainable_variables()
                    net.build_loss(
                        pixel_cls_labels  = b_pixel_cls_label,
                        pixel_cls_weights = b_pixel_cls_weight, 
                        pixel_link_labels = b_pixel_link_label, 
                        pixel_link_weights= b_pixel_link_weight,
                        anchor_gclasses       = b_gclasses,
                        anchor_glocalisations = b_glocalisations,
                        anchor_gscores        = b_scores,
                        do_summary        = do_summary)
                    
                    # gather losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) == 3
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixel_link_loss += total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss
                    
                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)
                    
    tf.summary.scalar('pixel_link_loss', pixel_link_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    
    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)
    
    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)
    
    train_ops = [apply_grad_op]
    
    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)
    
    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f'%(FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):# ema after updating
            train_ops.append(tf.group(ema_op))
         
    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op


def create_clones(batch_queue):
    if config.model_type == 'vgg16':
        from nets import pixel_link_symbol1 as pixel_link_symbol
    elif config.model_type == 'vgg16_dssd':
        from nets import pixel_link_symbol1 as pixel_link_symbol
    else:  # 'vgg16_dssd_ssd'
        from nets import pixel_link_symbol2 as pixel_link_symbol

    with tf.device('/cpu:0'):
        global_step = slim.create_global_step()
        learning_rate = tf.constant(FLAGS.learning_rate, name='learning_rate')
        optimizer = tf.train.MomentumOptimizer(learning_rate,
                                               momentum=FLAGS.momentum, name='Momentum')

        tf.summary.scalar('learning_rate', learning_rate)
    # place clones
    pixel_link_loss = 0  # for summary only
    gradients = []
    for clone_idx, gpu in enumerate(config.gpus):
        do_summary = clone_idx == 0  # only summary on the first clone
        reuse = clone_idx > 0
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
            with tf.name_scope(config.clone_scopes[clone_idx]) as clone_scope:
                with tf.device(gpu) as clone_device:
                    b_image, b_pixel_cls_label, b_pixel_cls_weight, \
                    b_pixel_link_label, b_pixel_link_weight = batch_queue.dequeue()
                    # build model and loss
                    net = pixel_link_symbol.PixelLinkNet(b_image, is_training=True)
                    net.build_loss(
                        pixel_cls_labels=b_pixel_cls_label,
                        pixel_cls_weights=b_pixel_cls_weight,
                        pixel_link_labels=b_pixel_link_label,
                        pixel_link_weights=b_pixel_link_weight,
                        do_summary=do_summary)

                    # gather losses
                    losses = tf.get_collection(tf.GraphKeys.LOSSES, clone_scope)
                    assert len(losses) == 2
                    total_clone_loss = tf.add_n(losses) / config.num_clones
                    pixel_link_loss += total_clone_loss

                    # gather regularization loss and add to clone_0 only
                    if clone_idx == 0:
                        regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                        total_clone_loss = total_clone_loss + regularization_loss

                    # compute clone gradients
                    clone_gradients = optimizer.compute_gradients(total_clone_loss)
                    gradients.append(clone_gradients)

    tf.summary.scalar('pixel_link_loss', pixel_link_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    # add all gradients together
    # note that the gradients do not need to be averaged, because the average operation has been done on loss.
    averaged_gradients = sum_gradients(gradients)

    apply_grad_op = optimizer.apply_gradients(averaged_gradients, global_step=global_step)

    train_ops = [apply_grad_op]

    bn_update_op = util.tf.get_update_op()
    if bn_update_op is not None:
        train_ops.append(bn_update_op)

    # moving average
    if FLAGS.using_moving_average:
        tf.logging.info('using moving average in training, \
        with decay = %f' % (FLAGS.moving_average_decay))
        ema = tf.train.ExponentialMovingAverage(FLAGS.moving_average_decay)
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([apply_grad_op]):  # ema after updating
            train_ops.append(tf.group(ema_op))

    train_op = control_flow_ops.with_dependencies(train_ops, pixel_link_loss, name='train_op')
    return train_op

def train(train_op):
    """
    def slim.learning.train(train_op,
          logdir,
          train_step_fn=train_step,
          train_step_kwargs=_USE_DEFAULT,
          log_every_n_steps=1,
          graph=None,
          master='',
          is_chief=True,
          global_step=None,
          number_of_steps=None,
          init_op=_USE_DEFAULT,
          init_feed_dict=None,
          local_init_op=_USE_DEFAULT,
          init_fn=None,
          ready_op=_USE_DEFAULT,
          summary_op=_USE_DEFAULT,
          save_summaries_secs=600,
          summary_writer=_USE_DEFAULT,
          startup_delay_steps=0,
          saver=None,
          save_interval_secs=600,
          sync_optimizer=None,
          session_config=None,
          session_wrapper=None,
          trace_every_n_steps=None,
          ignore_live_threads=False):
  Runs a training loop using a TensorFlow supervisor.
  When the sync_optimizer is supplied, gradient updates are applied
  synchronously. Otherwise, gradient updates are applied asynchronous.
  Args:
      train_op: A `Tensor` that, when executed, will apply the gradients and
      return the loss value.
    logdir: The directory where training logs are written to. If None, model
      checkpoints and summaries will not be written.
    train_step_fn: The function to call in order to execute a single gradient
      step. The function must have take exactly four arguments: the current
      session, the `train_op` `Tensor`, a global step `Tensor` and a dictionary.
    train_step_kwargs: A dictionary which is passed to the `train_step_fn`. By
      default, two `Boolean`, scalar ops called "should_stop" and "should_log"
      are provided.
    log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step are logged.
    graph: The graph to pass to the supervisor. If no graph is supplied the
      default graph is used.
    master: The address of the tensorflow master.
    is_chief: Specifies whether or not the training is being run by the primary
      replica during replica training.
    global_step: The `Tensor` representing the global step. If left as `None`,
      then training_util.get_or_create_global_step(), that is,
      tf.contrib.framework.global_step() is used.
    number_of_steps: The max number of gradient steps to take during training,
      as measured by 'global_step': training will stop if global_step is
      greater than 'number_of_steps'. If the value is left as None, training
      proceeds indefinitely.
    init_op: The initialization operation. If left to its default value, then
      the session is initialized by calling `tf.global_variables_initializer()`.
    init_feed_dict: A feed dictionary to use when executing the `init_op`.
    local_init_op: The local initialization operation. If left to its default
      value, then the session is initialized by calling
      `tf.local_variables_initializer()` and `tf.tables_initializer()`.
    init_fn: An optional callable to be executed after `init_op` is called. The
      callable must accept one argument, the session being initialized.
    ready_op: Operation to check if the model is ready to use. If left to its
      default value, then the session checks for readiness by calling
      `tf.report_uninitialized_variables()`.
    summary_op: The summary operation.
    save_summaries_secs: How often, in seconds, to save summaries.
    summary_writer: `SummaryWriter` to use.  Can be `None`
      to indicate that no summaries should be written. If unset, we
      create a SummaryWriter.
    startup_delay_steps: The number of steps to wait for before beginning. Note
      that this must be 0 if a sync_optimizer is supplied.
    saver: Saver to save checkpoints. If None, a default one will be created
      and used.
    save_interval_secs: How often, in seconds, to save the model to `logdir`.
    sync_optimizer: an instance of tf.train.SyncReplicasOptimizer, or a list of
      them. If the argument is supplied, gradient updates will be synchronous.
      If left as `None`, gradient updates will be asynchronous.
    session_config: An instance of `tf.ConfigProto` that will be used to
      configure the `Session`. If left as `None`, the default will be used.
    session_wrapper: A function that takes a `tf.Session` object as the only
      argument and returns a wrapped session object that has the same methods
      that the original object has, or `None`. Iff not `None`, the wrapped
      object will be used for training.
    trace_every_n_steps: produce and save a `Timeline` in Chrome trace format
      and add it to the summaries every `trace_every_n_steps`. If None, no trace
      information will be produced or saved.
    ignore_live_threads: If `True` ignores threads that remain running after
      a grace period when stopping the supervisor, instead of raising a
      RuntimeError.
  Returns:
    the value of the loss function after training.
    """
    summary_op = tf.summary.merge_all()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess_config.log_device_placement = False
    sess_config.allow_soft_placement = True

    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    
    init_fn = util.tf.get_init_fn(checkpoint_path = FLAGS.checkpoint_path,
                                  train_dir = FLAGS.train_dir,
                                  ignore_missing_vars = FLAGS.ignore_missing_vars,
                                  checkpoint_exclude_scopes = FLAGS.checkpoint_exclude_scopes)
    saver = tf.train.Saver(max_to_keep = 500, write_version = 2)
    slim.learning.train(
            train_op,
            logdir = FLAGS.train_dir,
            init_fn = init_fn,
            summary_op = summary_op,
            number_of_steps = FLAGS.max_number_of_steps,
            log_every_n_steps = FLAGS.log_every_n_steps,
            save_summaries_secs = 30,
            saver = saver,
            save_interval_secs = 1200,
            session_config = sess_config
    )


def main(_):
    # The choice of return dataset object via initialization method maybe confusing, 
    # but I need to print all configurations in this method, including dataset information. 
    dataset = config_initialization()   
    # embed()
    # pdb.set_trace()

    if config.model_type == 'vgg16_dssd_ssd':
        batch_queue = create_dataset_batch_queue2(dataset)
        train_op = create_clones2(batch_queue)
    else:
        batch_queue = create_dataset_batch_queue(dataset)
        train_op = create_clones(batch_queue)

    train(train_op)
    
    
if __name__ == '__main__':
    tf.app.run()
