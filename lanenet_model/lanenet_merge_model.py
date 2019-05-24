#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午5:28
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_merge_model.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet模型
"""
# from __future__ import print_function

import tensorflow as tf

from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import mobilenet_encoder
from encoder_decoder_model import fcn_decoder
from encoder_decoder_model import dense_encoder
from encoder_decoder_model import cnn_basenet
from lanenet_model import lanenet_discriminative_loss


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    实现语义分割模型
    """

    def __init__(self, phase, net_flag='vgg'):
        """

        """
        super(LaneNet, self).__init__()
        self._net_flag = net_flag
        self._phase = phase
        if self._net_flag == 'vgg':
            self._encoder = vgg_encoder.VGG16Encoder(phase=phase)
        elif self._net_flag == 'mobilenet':
            self._encoder = mobilenet_encoder.Mobilenet(phase=phase)
        elif self._net_flag == 'dense':
            self._encoder = dense_encoder.DenseEncoder(l=20, growthrate=8,
                                                       with_bc=True,
                                                       phase=phase,
                                                       n=5)
        self._decoder = fcn_decoder.FCNDecoder(phase=phase)
        return

    def __str__(self):
        """

        :return:
        """
        info = 'Semantic Segmentation use {:s} as basenet to encode'.format(self._net_flag)
        return info

    def _build_model(self, input_tensor, name):
        """
        前向传播过程
        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # first encode
            encode_ret = self._encoder.encode(input_tensor=input_tensor,
                                              name='encode')

            # second decode
            if self._net_flag.lower() == 'vgg':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['pool5',
                                                                     'pool4',
                                                                     'pool3'])
                return decode_ret
            elif self._net_flag.lower() == 'dense':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['Dense_Block_5',
                                                                     'Dense_Block_4',
                                                                     'Dense_Block_3'])
            elif self._net_flag.lower() == 'mobilenet':
                decode_ret = self._decoder.decode(input_tensor_dict=encode_ret,
                                                  name='decode',
                                                  decode_layer_list=['layer_18',
                                                                     'layer_14',
                                                                     'layer_7'])
                return decode_ret

    def compute_loss(self, input_tensor, binary_label, instance_label, ignore_label, name):
        """
        计算LaneNet模型损失函数
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')

            # 计算discriminative loss损失函数
            pix_embedding = inference_ret['deconv']
            # 像素嵌入


            # 计算二值分割损失函数
            decode_logits = inference_ret['logits']

            zeros = tf.zeros(tf.shape(binary_label))
            zeros = tf.cast(zeros, tf.int64)
            binary_label_f = tf.where(tf.equal(binary_label, ignore_label), zeros, binary_label)

            binary_label_plain = tf.reshape(
                binary_label_f,
                shape=[binary_label_f.get_shape().as_list()[0] *
                       binary_label_f.get_shape().as_list()[1] *
                       binary_label_f.get_shape().as_list()[2]])
            # 加入class weights
            unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_plain)
            counts = tf.cast(counts, tf.float32)
            inverse_weights = tf.divide(1.0, tf.log(tf.add(tf.divide(tf.constant(1.0), counts), tf.constant(1.02))))

            inverse_weights = tf.gather(inverse_weights, binary_label_f)
            zeros = tf.zeros(tf.shape(inverse_weights))
            inverse_weights = tf.where(binary_label == ignore_label, zeros, inverse_weights)

            binary_segmenatation_loss = tf.losses.sparse_softmax_cross_entropy(
                labels=binary_label_f, logits=decode_logits, weights=inverse_weights)
            binary_segmenatation_loss = tf.reduce_mean(binary_segmenatation_loss)

            # 计算discriminative loss
            image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])
            disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss.discriminative_loss(
                    pix_embedding, instance_label, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)

            # 合并损失
            if self._net_flag != "mobilenet":
                # bad way to do reg loss
                l2_reg_loss = tf.constant(0.0, tf.float32)
                for vv in tf.trainable_variables():
                    if 'bn' in vv.name:
                        continue
                    else:
                        l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
                l2_reg_loss *= 0.001
                total_loss = 0.5 * binary_segmenatation_loss + 0.5 * disc_loss + l2_reg_loss

            elif self._net_flag == "mobilenet":
                reg_losses = tf.contrib.slim.losses.get_regularization_losses()
                reg_loss_encode = tf.add_n(reg_losses, name="reg_loss_encode")

                decode_var_list = []
                for decode_var in tf.trainable_variables():
                    if 'decode' in decode_var.name:
                        decode_var_list.append(tf.nn.l2_loss(decode_var))
                reg_loss_decode = tf.add_n(decode_var_list)

                reg_loss = tf.add(reg_loss_encode, reg_loss_decode, name="reg_loss")

                tf.losses.add_loss(binary_segmenatation_loss, "binary_segmenatation_loss")
                tf.losses.add_loss(disc_loss, "disc_loss")
                tf.losses.add_loss(reg_loss, "reg_loss")

                total_loss = 0.6 * binary_segmenatation_loss + 0.4 * disc_loss + reg_loss * 0.001
                tf.losses.add_loss(total_loss, "total_loss")

            # tf.Print(total_loss, [tf.shape(total_loss)], message="total_loss: ")

            ret = {
                'total_loss': total_loss,
                'binary_seg_logits': decode_logits,
                'instance_seg_logits': pix_embedding,
                'binary_seg_loss': binary_segmenatation_loss,
                'discriminative_loss': disc_loss
            }

            return ret

    def inference(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            # 前向传播获取logits
            inference_ret = self._build_model(input_tensor=input_tensor, name='inference')
            # 计算二值分割损失函数

            # 计算像素嵌入
            pix_embedding = inference_ret['deconv']

            # 像素嵌入
            decode_logits = inference_ret['logits']

            prob_seg_ret = tf.nn.softmax(logits=decode_logits)
            binary_seg_ret = tf.argmax(prob_seg_ret, axis=-1)

            return binary_seg_ret, pix_embedding, prob_seg_ret


if __name__ == '__main__':
    model = LaneNet(tf.constant('train', dtype=tf.string))
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input')
    binary_label = tf.placeholder(dtype=tf.int64, shape=[1, 256, 512, 1], name='label')
    instance_label = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 1], name='label')
    ret = model.compute_loss(input_tensor=input_tensor, binary_label=binary_label,
                             instance_label=instance_label, name='loss')
    for vv in tf.trainable_variables():
        if 'bn' in vv.name:
            continue
        print(vv.name)
