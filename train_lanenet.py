#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-18 下午7:31
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet.py
# @IDE: PyCharm Community Edition
"""
训练lanenet模型
"""
import argparse
import math
import os
import os.path as ops
import time

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from config import global_config
from data_provider import lanenet_data_processor
from lanenet_model import lanenet_merge_model

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 600,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 60,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', '/home/remusm/projects/laneNet/dataset/', 'Where the dataset reside.')
flags.DEFINE_string('weights_path', '', 'Where the model reside.')
flags.DEFINE_string('net', 'aspp_mobilenet', 'Where the dataset reside.')
flags.DEFINE_string('my_checkpoint', 'true', 'If is loaded mine checkpoint, different var names.')


# os.environ['CUDA_VISIBLE_DEVICES'] = ''


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', type=str, help='The training dataset dir path')
    parser.add_argument('--net', type=str, help='Which base net work to use', default='aspp_mobilenet')
    parser.add_argument('--weights_path', type=str, help='The pretrained weights path')
    parser.add_argument('--my_checkpoint', type=str,
                        help='If the checkpoints is saved by me or not (different variable names)', default="true")
    parser.add_argument('--model_save_dir', type=str, help='model dir',
                        default='./model/AVM_preTS_new_arch_ASPP_3')
    parser.add_argument('--tboard_save_dir', type=str, help='tboard dir',
                        default='./tboard/AVM_preTS_new_arch_ASPP_3')
    parser.add_argument('--ignore_labels_path', type=str, help='path to ignore labels mask',
                        default='./ignore_labels_AVM.png')

    return parser.parse_args()


def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def train_net(dataset_dir, weights_path=None, net_flag='vgg', save_dir="./logs/train/lanenet",
              tboard_save_path="./tboard/lanenet",
              ignore_labels_path="/media/remus/datasets/AVMSnapshots/AVM/ignore_labels.png",
              my_checkpoint="true"):
    """

    :param save_dir:
    :param ignore_labels_path:
    :param tboard_save_path:
    :param dataset_dir:
    :param net_flag: choose which base network to use
    :param weights_path:
    :return:
    """
    train_dataset_file = ops.join(dataset_dir, 'train.txt')
    val_dataset_file = ops.join(dataset_dir, 'val.txt')

    assert ops.exists(train_dataset_file)
    # tf.enable_eager_execution()

    train_dataset = lanenet_data_processor.DataSet(train_dataset_file)
    val_dataset = lanenet_data_processor.DataSet(val_dataset_file)

    with tf.device('/gpu:1'):
        input_tensor = tf.placeholder(dtype=tf.float32,
                                      shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                             CFG.TRAIN.IMG_WIDTH, 3],
                                      name='input_tensor')
        binary_label_tensor = tf.placeholder(dtype=tf.int64,
                                             shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                    CFG.TRAIN.IMG_WIDTH, 1],
                                             name='binary_input_label')
        instance_label_tensor = tf.placeholder(dtype=tf.float32,
                                               shape=[CFG.TRAIN.BATCH_SIZE, CFG.TRAIN.IMG_HEIGHT,
                                                      CFG.TRAIN.IMG_WIDTH],
                                               name='instance_input_label')
        phase = tf.placeholder(dtype=tf.string, shape=None, name='net_phase')

        net = lanenet_merge_model.LaneNet(net_flag=net_flag, phase=phase)

        # calculate the loss
        compute_ret = net.compute_loss(input_tensor=input_tensor, binary_label=binary_label_tensor,
                                       instance_label=instance_label_tensor, ignore_label=255, name='lanenet_model')
        total_loss = compute_ret['total_loss']
        binary_seg_loss = compute_ret['binary_seg_loss']
        disc_loss = compute_ret['discriminative_loss']
        pix_embedding = compute_ret['instance_seg_logits']

        # calculate the accuracy
        out_logits = compute_ret['binary_seg_logits']
        out_logits = tf.nn.softmax(logits=out_logits)
        out_logits_out = tf.argmax(out_logits, axis=-1)
        out = tf.argmax(out_logits, axis=-1)
        out = tf.expand_dims(out, axis=-1)

        idx = tf.where(tf.equal(binary_label_tensor, 1))
        pix_cls_ret = tf.gather_nd(out, idx)
        accuracy = tf.count_nonzero(pix_cls_ret)
        accuracy = tf.divide(accuracy, tf.cast(tf.shape(pix_cls_ret)[0], tf.int64))

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(CFG.TRAIN.LEARNING_RATE, global_step,
                                                   CFG.TRAIN.LR_DECAY_STEPS, CFG.TRAIN.LR_DECAY_RATE, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9).minimize(loss=total_loss,
                                                                    var_list=tf.trainable_variables(),
                                                                    global_step=global_step)
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Set tf saver
    from correct_path_saver import restore_from_classification_checkpoint_fn, get_variables_available_in_checkpoint

    if my_checkpoint == "true":
        # init_saver = tf.train.Saver()
        available_var_map = (get_variables_available_in_checkpoint(
            tf.global_variables(), weights_path, include_global_step=True))

        init_saver = tf.train.Saver(available_var_map)
    else:
        if weights_path is not None:
            var_map = restore_from_classification_checkpoint_fn("lanenet_model/inference")
            available_var_map = (get_variables_available_in_checkpoint(
                var_map, weights_path, include_global_step=True))

            init_saver = tf.train.Saver(available_var_map)
        else:
            init_saver = tf.train.Saver()

    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = '{:s}_lanenet_{:s}.ckpt'.format(net_flag, str(train_start_time))
    model_save_path = ops.join(save_dir, model_name)

    # Set tf summary
    if not ops.exists(tboard_save_path):
        os.makedirs(tboard_save_path)
    train_cost_scalar = tf.summary.scalar(name='train_cost', tensor=total_loss)
    val_cost_scalar = tf.summary.scalar(name='val_cost', tensor=total_loss)
    train_accuracy_scalar = tf.summary.scalar(name='train_accuracy', tensor=accuracy)
    val_accuracy_scalar = tf.summary.scalar(name='val_accuracy', tensor=accuracy)
    train_binary_seg_loss_scalar = tf.summary.scalar(name='train_binary_seg_loss', tensor=binary_seg_loss)
    val_binary_seg_loss_scalar = tf.summary.scalar(name='val_binary_seg_loss', tensor=binary_seg_loss)
    train_instance_seg_loss_scalar = tf.summary.scalar(name='train_instance_seg_loss', tensor=disc_loss)
    val_instance_seg_loss_scalar = tf.summary.scalar(name='val_instance_seg_loss', tensor=disc_loss)
    learning_rate_scalar = tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    train_merge_summary_op = tf.summary.merge([train_accuracy_scalar, train_cost_scalar,
                                               learning_rate_scalar, train_binary_seg_loss_scalar,
                                               train_instance_seg_loss_scalar])
    val_merge_summary_op = tf.summary.merge([val_accuracy_scalar, val_cost_scalar,
                                             val_binary_seg_loss_scalar, val_instance_seg_loss_scalar])

    # Set sess configuration
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'
    # sess_config.device_count = {'GPU': 0}

    sess = tf.Session(config=sess_config)
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess=sess,
    #                                                grpc_debug_server_addresses="remusm-pc:7000",
    #                                                send_traceback_and_source_code=False)

    summary_writer = tf.summary.FileWriter(tboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = CFG.TRAIN.EPOCHS

    tf.logging.info('Global configuration is as follows:')
    tf.logging.info(CFG)

    iter_saver = tf.train.Saver(max_to_keep=10)
    best_saver = tf.train.Saver(max_to_keep=3)

    with sess.as_default():

        sess.run(tf.global_variables_initializer())

        tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='',
                             name='{:s}/lanenet_model.pb'.format(save_dir))

        if weights_path is None:
            tf.logging.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            tf.logging.info('Restore model from last model checkpoint {:s}'.format(weights_path))
            init_saver.restore(sess=sess, save_path=weights_path)

            assign_op = global_step.assign(0)
            sess.run(assign_op)

        # 加载预训练参数
        if net_flag == 'vgg' and weights_path is None:
            pretrained_weights = np.load(
                './data/vgg16.npy',
                encoding='latin1').item()

            for vv in tf.trainable_variables():
                weights_key = vv.name.split('/')[-3]
                try:
                    weights = pretrained_weights[weights_key][0]
                    _op = tf.assign(vv, weights)
                    sess.run(_op)
                except Exception as e:
                    continue

        train_cost_time_mean = []
        val_cost_time_mean = []
        ignore_label_mask = cv2.imread(ignore_labels_path)
        last_c = 100000

        for epoch in range(train_epochs):
            # training part
            t_start = time.time()

            with tf.device('/cpu:0'):
                gt_imgs, binary_gt_labels, instance_gt_labels = train_dataset.next_batch(
                    CFG.TRAIN.BATCH_SIZE,
                    ignore_label_mask=ignore_label_mask,
                    ignore_label=255)

                # gt_imgs = [tmp - VGG_MEAN for tmp in gt_imgs]
                # gt_imgs = [tmp / 128.0 - 1.0 for tmp in gt_imgs]

                binary_gt_labels = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels]

            phase_train = 'train'

            _, c, train_accuracy, train_summary, binary_loss, instance_loss, embedding, binary_seg_img, g_step = \
                sess.run([optimizer, total_loss,
                          accuracy,
                          train_merge_summary_op,
                          binary_seg_loss,
                          disc_loss,
                          pix_embedding,
                          out_logits_out,
                          global_step],
                         feed_dict={input_tensor: gt_imgs,
                                    binary_label_tensor: binary_gt_labels,
                                    instance_label_tensor: instance_gt_labels,
                                    phase: phase_train})
            # if epoch % 10 == 0:
            # tf.logging.info("Epoch {}."
            #     "Total loss: {}. Train acc: {}."
            #     " Binary loss: {}. Instance loss: {}".format(epoch, c, train_accuracy,
            #                                                  binary_loss, instance_loss))

            if math.isnan(c) or math.isnan(binary_loss) or math.isnan(instance_loss):
                tf.logging.error('cost is: {:.5f}'.format(c))
                tf.logging.error('binary cost is: {:.5f}'.format(binary_loss))
                tf.logging.error('instance cost is: {:.5f}'.format(instance_loss))
                # cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('nan_image.png', gt_imgs[0])
                cv2.imwrite('nan_instance_label.png', instance_gt_labels[0])
                cv2.imwrite('nan_binary_label.png', binary_gt_labels[0] * 255)
                return

            if epoch % 100 == 0:
                # cv2.imwrite('nan_image.png', gt_imgs[0] + VGG_MEAN)
                cv2.imwrite('image.png', gt_imgs[0])
                cv2.imwrite('binary_label.png', binary_gt_labels[0] * 255)
                cv2.imwrite('instance_label.png', instance_gt_labels[0])
                cv2.imwrite('binary_seg_img.png', binary_seg_img[0] * 255)

                for i in range(4):
                    embedding[0][:, :, i] = minmax_scale(embedding[0][:, :, i])
                embedding_image = np.array(embedding[0], np.uint8)
                cv2.imwrite('embedding.png', embedding_image[:, :, :-1])

            cost_time = time.time() - t_start
            train_cost_time_mean.append(cost_time)
            summary_writer.add_summary(summary=train_summary, global_step=epoch)

            # validation part
            with tf.device('/cpu:0'):
                gt_imgs_val, binary_gt_labels_val, instance_gt_labels_val \
                    = val_dataset.next_batch(CFG.TRAIN.VAL_BATCH_SIZE, ignore_label_mask=ignore_label_mask)

                # gt_imgs_val = [tmp - VGG_MEAN for tmp in gt_imgs_val]
                # gt_imgs_val = [tmp / 128.0 - 1.0 for tmp in gt_imgs_val]

                binary_gt_labels_val = [np.expand_dims(tmp, axis=-1) for tmp in binary_gt_labels_val]
            phase_val = 'test'

            t_start_val = time.time()
            c_val, val_summary, val_accuracy, val_binary_seg_loss, val_instance_seg_loss = \
                sess.run([total_loss, val_merge_summary_op, accuracy, binary_seg_loss, disc_loss],
                         feed_dict={input_tensor: gt_imgs_val,
                                    binary_label_tensor: binary_gt_labels_val,
                                    instance_label_tensor: instance_gt_labels_val,
                                    phase: phase_val})

            if epoch % 100 == 0:
                # cv2.imwrite('test_image.png', gt_imgs_val[0] + VGG_MEAN)
                cv2.imwrite('test_image.png', (gt_imgs_val[0] + 1.0) * 128)

            summary_writer.add_summary(val_summary, global_step=epoch)

            cost_time_val = time.time() - t_start_val
            val_cost_time_mean.append(cost_time_val)

            if epoch % CFG.TRAIN.DISPLAY_STEP == 0:
                tf.logging.info(
                    'Step: {:d} total_loss= {:6f} binary_seg_loss= {:6f} instance_seg_loss= {:6f} accuracy= {:6f}'
                    ' mean_cost_time= {:5f}s '.
                        format(epoch + 1, c, binary_loss, instance_loss, train_accuracy,
                               np.mean(train_cost_time_mean)))
                train_cost_time_mean.clear()

            if epoch % CFG.TRAIN.TEST_DISPLAY_STEP == 0:
                tf.logging.info('Step_Val: {:d} total_loss= {:6f} binary_seg_loss= {:6f} '
                                'instance_seg_loss= {:6f} accuracy= {:6f} '
                                'mean_cost_time= {:5f}s '.
                                format(epoch + 1, c_val, val_binary_seg_loss, val_instance_seg_loss, val_accuracy,
                                       np.mean(val_cost_time_mean)))
                val_cost_time_mean.clear()

            if epoch % 2000 == 0:
                iter_saver.save(sess=sess, save_path=model_save_path, global_step=epoch)

                if c < last_c:
                    last_c = c
                    save_dir_best = save_dir + "/best"
                    if not ops.exists(save_dir_best):
                        os.makedirs(save_dir_best)
                    best_model_save_path = ops.join(save_dir_best, model_name)

                    best_saver.save(sess=sess, save_path=best_model_save_path, global_step=epoch)

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    # train lanenet
    train_net(args.dataset_dir, args.weights_path, net_flag=args.net, save_dir=args.model_save_dir,
              tboard_save_path=args.tboard_save_dir, ignore_labels_path=args.ignore_labels_path,
              my_checkpoint=args.my_checkpoint)
