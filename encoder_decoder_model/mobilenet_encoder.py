from collections import OrderedDict

import tensorflow as tf
import tensorflow.contrib.slim as slim
from slim.nets.mobilenet import mobilenet_v2

from encoder_decoder_model import cnn_basenet


class Mobilenet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        """

        :param phase:
        """
        super(Mobilenet, self).__init__()
        self._train_phase = tf.constant('train', dtype=tf.string)
        self._test_phase = tf.constant('test', dtype=tf.string)
        self._phase = phase
        self._is_training = self._init_phase()

    def _init_phase(self):
        """

        :return:
        """
        return tf.equal(self._phase, self._train_phase)

    def encode(self, input_tensor, name):
        """
        根据vgg16框架对输入的tensor进行编码
        :param input_tensor:
        :param name:
        :param flags:
        :return: 输出vgg16编码特征
        """

        # print(self._phase)
        # model_path = '/logs/'
        # input_tensor = tf.Print(input_tensor, [tf.reduce_sum(tf.to_int32(tf.is_nan(input_tensor))) > 0],
        #                         message="input_tensor")

        with slim.arg_scope(mobilenet_v2.training_scope(is_training=self._is_training)):
            logits, endpoints = mobilenet_v2.mobilenet_base(
                input_tensor=input_tensor,
                is_training=self._is_training)

            ret = OrderedDict()

            ret['layer_7'] = dict()
            # asymetric_7 = self.conv2d(inputdata=endpoints["layer_7"], out_channel=32,
            #                           kernel_size=[3, 1], use_bias=False, name='asymetric_7')
            ret['layer_7']['data'] = endpoints["layer_7"]
            ret['layer_7']['shape'] = endpoints["layer_7"].get_shape().as_list()

            ret['layer_14'] = dict()
            # asymetric_14 = self.conv2d(inputdata=endpoints["layer_14"], out_channel=96,
            #                            kernel_size=[3, 1], use_bias=False, name='asymetric_14')
            ret['layer_14']['data'] = endpoints["layer_14"]
            ret['layer_14']['shape'] = endpoints["layer_14"].get_shape().as_list()

            ret['layer_18'] = dict()
            # asymetric_19 = self.conv2d(inputdata=endpoints["layer_19"], out_channel=1280,
            #                            kernel_size=[3, 1], use_bias=False, name='asymetric_19')
            ret['layer_18']['data'] = endpoints["layer_18"]
            ret['layer_18']['shape'] = endpoints["layer_18"].get_shape().as_list()

        return ret
