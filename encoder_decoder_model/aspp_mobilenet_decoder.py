#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-29 下午2:38
# @Author  : Luo Yao
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : fcn_decoder.py
# @IDE: PyCharm Community Edition
"""
实现一个全卷积网络解码类
"""
import tensorflow as tf

from encoder_decoder_model import cnn_basenet
from encoder_decoder_model import vgg_encoder
from encoder_decoder_model import dense_encoder


class ASPP_Decoder(cnn_basenet.CNNBaseModel):
    """
    实现一个全卷积解码类
    """

    def __init__(self, phase):
        """

        """
        super(ASPP_Decoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def decode(self, input_tensor_dict, name):
        """
        解码特征信息反卷积还原
        :param input_tensor_dict:
        :param decode_layer_list: 需要解码的层名称需要由深到浅顺序写
                                  eg. ['pool5', 'pool4', 'pool3']
        :param name:
        :return:
        """
        ret = dict()

        with tf.variable_scope(name):
            # deconv_final = self.deconv2d(inputdata=input_tensor_dict, out_channel=64, kernel_size=8,
            #                              stride=4, use_bias=False, name='deconv_final')

            from deeplab_util.model import _resize_bilinear
            resize_final = _resize_bilinear(input_tensor_dict, [288, 416])
            deconv_final = self.conv2d(inputdata=resize_final, out_channel=64,
                                      kernel_size=1, use_bias=False, name='deconv_final')

            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')
            ret['prev_logits'] = score_final
            pix_embedding = self.conv2d(inputdata=deconv_final, out_channel=4, kernel_size=1,
                                        use_bias=False, name='pix_embedding_conv')
            pix_embedding = self.relu(inputdata=pix_embedding, name='pix_embedding_relu')
            pix_embedding_dub = tf.identity(pix_embedding, name="pix_embedding_dub")
            tf.stop_gradient(pix_embedding_dub)

            combined_layers = tf.concat([deconv_final, pix_embedding_dub, score_final], axis=-1)
            score_final = self.conv2d(inputdata=combined_layers, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final_combined')

            ret['logits'] = score_final
            ret['deconv'] = pix_embedding

        return ret
