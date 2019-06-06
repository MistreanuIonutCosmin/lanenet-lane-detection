from collections import OrderedDict
import collections
import tensorflow as tf
import common
from encoder_decoder_model.deeplab_util import model
import config.global_config as config
import tensorflow.contrib.slim as slim
from slim.nets.mobilenet import mobilenet_v2

from encoder_decoder_model import cnn_basenet

CFG = config.cfg


class ASPP_Mobilenet(cnn_basenet.CNNBaseModel):

    def __init__(self, phase):
        """

        :param phase:
        """
        super(ASPP_Mobilenet, self).__init__()
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
        model_options = common.ModelOptions(
            outputs_to_num_classes=[2],
            crop_size=[CFG.TRAIN.IMG_HEIGHT, CFG.TRAIN.IMG_WIDTH],
            atrous_rates=CFG.TRAIN.ATROUS_RATES,
            output_stride=CFG.TRAIN.OUTPUT_STRIDE)

        outputs_to_scales_to_logits = model.multi_scale_logits(input_tensor,
                                                               model_options,
                                                               None,
                                                               weight_decay=0.0001,
                                                               is_training=False,
                                                               fine_tune_batch_norm=False)

        return outputs_to_scales_to_logits
