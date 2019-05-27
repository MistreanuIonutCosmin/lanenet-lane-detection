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


class FCNDecoder(cnn_basenet.CNNBaseModel):
    """
    实现一个全卷积解码类
    """

    def __init__(self, phase):
        """

        """
        super(FCNDecoder, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def decode(self, input_tensor_dict, decode_layer_list, name):
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
            # score stage 1
            input_tensor = input_tensor_dict[decode_layer_list[0]]['data']

            score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                kernel_size=1, use_bias=False, name='score_origin')
            # score = tf.Print(score, [tf.reduce_sum(tf.to_int32(tf.is_nan(input_tensor))) > 0,
            #                          tf.reduce_sum(tf.to_int32(tf.is_nan(score))) > 0],
            #                  message='input, score - {:s}: '.format(decode_layer_list[0]))

            decode_layer_list = decode_layer_list[1:]
            for i in range(len(decode_layer_list)):
                deconv = self.deconv2d(inputdata=score, out_channel=64, kernel_size=4,
                                       stride=2, use_bias=False, name='deconv_{:d}'.format(i + 1))

                input_tensor = input_tensor_dict[decode_layer_list[i]]['data']
                score = self.conv2d(inputdata=input_tensor, out_channel=64,
                                    kernel_size=1, use_bias=False, name='score_{:d}'.format(i + 1))

                fused = tf.add(deconv, score, name='fuse_{:d}'.format(i + 1))

                score = fused

            deconv_final = self.deconv2d(inputdata=score, out_channel=64, kernel_size=16,
                                         stride=8, use_bias=False, name='deconv_final')

            score_final = self.conv2d(inputdata=deconv_final, out_channel=2,
                                      kernel_size=1, use_bias=False, name='score_final')

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


if __name__ == '__main__':
    vgg_encoder = vgg_encoder.VGG16Encoder(phase=tf.constant('train', tf.string))
    dense_encoder = dense_encoder.DenseEncoder(l=40, growthrate=12,
                                               with_bc=True, phase='train', n=5)
    decoder = FCNDecoder(phase='train')

    in_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 256, 512, 3],
                               name='input')

    vgg_encode_ret = vgg_encoder.encode(in_tensor, name='vgg_encoder')
    dense_encode_ret = dense_encoder.encode(in_tensor, name='dense_encoder')
    decode_ret = decoder.decode(vgg_encode_ret, name='decoder',
                                decode_layer_list=['pool5',
                                                   'pool4',
                                                   'pool3'])
