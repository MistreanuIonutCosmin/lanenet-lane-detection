#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
测试LaneNet模型
"""
import os
import os.path as ops
import argparse
import time
import math

import tensorflow as tf
import glob
import glog as log
import numpy as np
import matplotlib.pyplot as plt
import cv2

from lanenet_model import lanenet_merge_model
from lanenet_model import lanenet_cluster
from lanenet_model import lanenet_postprocess
from config import global_config
from correct_path_saver import restore_from_classification_checkpoint_fn, get_variables_available_in_checkpoint

CFG = global_config.cfg
VGG_MEAN = [103.939, 116.779, 123.68]
# os.environ["CUDA_VISIBLE_DEVICES"]=''


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='The image path or the src image save dir',
                        default='/media/remus/datasets/AVMSnapshots/AVM/val_images')
    # default = '/media/remus/datasets/AVMSnapshots/AVM/val_images/0021_AVMFrontCamera.png')
    parser.add_argument('--weights_path', type=str, help='The model weights path',
                        default='/home/remusm/projects/laneNet/model/mobilenet_preTuS_combined/best/mobilenet_lanenet_2019-05-23-15-38-03.ckpt-144000')
                        # default='/media/remus/projects/lanenet-lane-detection/weights/AVM_ignore_label/tusimple_lanenet_vgg_2019-03-28-15-42-02.ckpt-200000')
    parser.add_argument('--encoder', type=str, help='If use gpu set 1 or 0 instead', default="mobilenet")
    parser.add_argument('--is_batch', type=str, help='If test a batch of images', default='true')
    parser.add_argument('--batch_size', type=int, help='The batch size of the test images', default=4)
    parser.add_argument('--save_dir', type=str, help='Test result image save dir',
                        default='/media/remus/datasets/AVMSnapshots/AVM/mobilenet_pTuS_combined/')
    parser.add_argument('--use_gpu', type=int, help='If use gpu set 1 or 0 instead', default=1)

    return parser.parse_args()


def minmax_scale(input_arr):
    """virtual

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(image_path, weights_path, use_gpu, save_dir):
    """

    :param save_dir:
    :param image_path:
    :param weights_path:
    :param use_gpu:
    :return:
    """
    assert ops.exists(image_path), '{:s} not exist'.format(image_path)

    log.info('开始读取图像数据并进行预处理')
    t_start = time.time()
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 128.0 - 1.0
    log.info('图像读取完毕, 耗时: {:.5f}s'.format(time.time() - t_start))

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag='mobilenet')
    binary_seg_ret, instance_seg_ret, _ = net.inference(input_tensor=input_tensor, name='lanenet_model')
    binary_seg_ret_32 = tf.cast(binary_seg_ret, tf.int32, name="binary_seg")

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    # Set tf saver
    # if weights_path is not None:
    #     var_map = restore_from_classification_checkpoint_fn("lanenet_model/inference")
    #     available_var_map = (get_variables_available_in_checkpoint(
    #         var_map, weights_path, include_global_step=False))
    #
    #     saver = tf.train.Saver(available_var_map)
    saver = tf.train.Saver()
    iter_saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 1})
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret_32, instance_seg_ret],
                                                        feed_dict={input_tensor: [image]})

        t_cost = time.time() - t_start
        log.info('单张图像车道线预测耗时: {:.5f}s'.format(t_cost))

        binary_seg_image[0] = postprocessor.postprocess(binary_seg_image[0])
        mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image[0],
                                           instance_seg_ret=instance_seg_image[0])

        for i in range(4):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)

        plt.figure('mask_image')
        plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(image_vis[:, :, (2, 1, 0)])
        plt.figure('instance_image')
        plt.imshow(embedding_image[:, :, (3, 1, 0)])
        plt.figure('binary_image')
        plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show()

        mask_image = mask_image[:, :, (2, 1, 0)]
        image_name = ops.split(image_path)[1]
        image_save_path = ops.join(save_dir, image_name)
        cv2.imwrite(image_save_path, mask_image)

        iter_saver.save(sess=sess, save_path=save_dir + "inference_models/model20.ckpt")
        tf.train.write_graph(sess.graph.as_graph_def(), save_dir + "inference_models/", "graph20.pb")
        # tf.train.write_graph(graph_or_graph_def=sess.graph, logdir='', name='{:s}/lanenet_model.pb'.format(save_dir))

    sess.close()

    return


def test_lanenet_batch(image_dir, weights_path, batch_size, use_gpu, save_dir=None, encoder="vgg"):
    """

    :param image_dir:
    :param weights_path:
    :param batch_size:
    :param use_gpu:
    :param save_dir:
    :return:
    """
    assert ops.exists(image_dir), '{:s} not exist'.format(image_dir)

    log.info('开始获取图像文件路径...')
    image_path_list = glob.glob('{:s}/**/*.jpg'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.png'.format(image_dir), recursive=True) + \
                      glob.glob('{:s}/**/*.jpeg'.format(image_dir), recursive=True)

    input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3], name='input_tensor')
    phase_tensor = tf.constant('test', tf.string)

    net = lanenet_merge_model.LaneNet(phase=phase_tensor, net_flag=encoder)
    binary_seg_ret, instance_seg_ret, prob_seg_ret = net.inference(input_tensor=input_tensor, name='lanenet_model')

    cluster = lanenet_cluster.LaneNetCluster()
    postprocessor = lanenet_postprocess.LaneNetPoseProcessor()

    # Set tf saver
    # if weights_path is not None:
    #     var_map = restore_from_classification_checkpoint_fn("")
    #     available_var_map = (get_variables_available_in_checkpoint(
    #         var_map, weights_path, include_global_step=False))
    #
    #     saver = tf.train.Saver(available_var_map)
    saver = tf.train.Saver()

    # Set sess configuration
    if use_gpu:
        sess_config = tf.ConfigProto(device_count={'GPU': 1})
        # sess_config = tf.ConfigProto(device_count={'CPU': 0})

    else:
        sess_config = tf.ConfigProto(device_count={'CPU': 1})
        # sess_config = tf.ConfigProto(device_count={'GPU': 0})

    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    ignore_labels = cv2.imread('/media/remus/datasets/AVMSnapshots/AVM/ignore_labels.png')
    ignore_labels = cv2.cvtColor(ignore_labels, cv2.COLOR_BGR2GRAY)

    sess = tf.Session(config=sess_config)

    with sess.as_default():

        saver.restore(sess=sess, save_path=weights_path)

        epoch_nums = int(math.ceil(len(image_path_list) / batch_size))

        for epoch in range(epoch_nums):
            log.info('[Epoch:{:d}] starts image reading and preprocessing...'.format(epoch))
            t_start = time.time()
            image_path_epoch = image_path_list[epoch * batch_size:(epoch + 1) * batch_size]
            image_list_epoch = [cv2.imread(tmp, cv2.IMREAD_COLOR) for tmp in image_path_epoch]
            image_vis_list = image_list_epoch
            image_list_epoch = [cv2.resize(tmp, (512, 256), interpolation=cv2.INTER_LINEAR)
                                for tmp in image_list_epoch]

            if encoder == "mobilenet":
                image_list_epoch = [tmp / 128.0 - 1.0 for tmp in image_list_epoch]
            else:
                image_list_epoch = [tmp - VGG_MEAN for tmp in image_list_epoch]
            t_cost = time.time() - t_start
            log.info(
                '[Epoch:{:d}] preprocesses {:d} images, total time: {:.5f}s, average time per sheet: {:.5f}'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            t_start = time.time()
            binary_seg_images, instance_seg_images, prob_seg_images = sess.run(
                [binary_seg_ret, instance_seg_ret, prob_seg_ret], feed_dict={input_tensor: image_list_epoch})
            t_cost = time.time() - t_start
            log.info(
                '[Epoch:{:d}] predicts {:d} image lane lines, total time: {:.5f}s, average time per sheet: {:.5f}s'.format(
                    epoch, len(image_path_epoch), t_cost, t_cost / len(image_path_epoch)))

            cluster_time = []
            for index, binary_seg_image in enumerate(binary_seg_images):
                t_start = time.time()
                binary_seg_image[ignore_labels == 0] = 0
                binary_seg_image = postprocessor.postprocess(binary_seg_image)
                mask_image = cluster.get_lane_mask(binary_seg_ret=binary_seg_image,
                                                   instance_seg_ret=instance_seg_images[index])
                cluster_time.append(time.time() - t_start)
                mask_image = cv2.resize(mask_image, (image_vis_list[index].shape[1],
                                                     image_vis_list[index].shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

                _instance_seg_images = np.copy(instance_seg_images)
                prob_seg_image = prob_seg_images[index, :, :, 1]

                for i in range(4):
                    _instance_seg_images[index][:, :, i] = minmax_scale(instance_seg_images[index][:, :, i])
                    _embedding_image = np.array(_instance_seg_images[index], np.uint8)

                if save_dir is None:
                    plt.ion()
                    plt.figure('mask_image')
                    plt.imshow(mask_image[:, :, (2, 1, 0)])
                    plt.figure('src_image')
                    plt.imshow(image_vis_list[index][:, :, (2, 1, 0)])
                    plt.pause(3.0)
                    plt.show()
                    plt.ioff()

                if save_dir is not None:
                    mask_image = cv2.addWeighted(image_vis_list[index], 1.0, mask_image, 1.0, 0)
                    image_name = ops.split(image_path_epoch[index])[1]
                    image_save_path = ops.join(save_dir + "/image", image_name)
                    mask_save_path = ops.join(save_dir + "/mask", image_name)
                    prob_save_path = ops.join(save_dir + "/prob", image_name)
                    embedding_save_path = ops.join(save_dir + "/embedding", image_name)
                    cv2.imwrite(mask_save_path, binary_seg_image * 255)
                    cv2.imwrite(prob_save_path,  prob_seg_image * 255)
                    cv2.imwrite(image_save_path, mask_image)
                    cv2.imwrite(embedding_save_path, _embedding_image[:, :, (2, 1, 0)])
                    # cv2.imwrite(embedding_save_path + "_", _embedding_image[:, :, (3, 2, 1)])

            log.info(
                '[Epoch:{:d}] performs {:d} image lane line clustering, which takes a total of time: {:.5f}s, average time per sheet: {:.5f}'.format(
                    epoch, len(image_path_epoch), np.sum(cluster_time), np.mean(cluster_time)))

    sess.close()

    return


if __name__ == '__main__':
    # init args
    args = init_args()

    if args.save_dir is not None and not ops.exists(args.save_dir):
        log.error('{:s} not exist and has been made'.format(args.save_dir))
        os.makedirs(args.save_dir)

    image_save_path = ops.join(args.save_dir + "/image")
    mask_save_path = ops.join(args.save_dir + "/mask")
    prob_save_path = ops.join(args.save_dir + "/prob")
    embedding_save_path = ops.join(args.save_dir + "/embedding")
    if not ops.exists(image_save_path):
        os.makedirs(image_save_path)
        os.makedirs(mask_save_path)
        os.makedirs(prob_save_path)
        os.makedirs(embedding_save_path)

    if args.is_batch.lower() == 'false':
        # test hnet model on single image
        test_lanenet(args.image_path, args.weights_path, args.use_gpu, save_dir=args.save_dir)
    else:
        # test hnet model on a batch of image
        test_lanenet_batch(image_dir=args.image_path, weights_path=args.weights_path,
                           save_dir=args.save_dir, use_gpu=args.use_gpu, batch_size=args.batch_size,
                           encoder=args.encoder)
