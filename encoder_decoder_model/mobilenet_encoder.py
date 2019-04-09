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

        with slim.arg_scope(mobilenet_v2.training_scope(is_training=self._is_training)):
            logits, endpoints = mobilenet_v2.mobilenet_base(
                input_tensor=input_tensor,
                is_training=self._is_training)

            ret = OrderedDict()

            ret['layer_7'] = dict()
            ret['layer_7']['data'] = endpoints["layer_7/output"]
            ret['layer_7']['shape'] = endpoints["layer_7/output"].get_shape().as_list()

            ret['layer_14'] = dict()
            ret['layer_14']['data'] = endpoints["layer_14/output"]
            ret['layer_14']['shape'] = endpoints["layer_14/output"].get_shape().as_list()

            ret['layer_18'] = dict()
            ret['layer_18']['data'] = endpoints["layer_18/output"]
            ret['layer_18']['shape'] = endpoints["layer_18/output"].get_shape().as_list()

        return ret
