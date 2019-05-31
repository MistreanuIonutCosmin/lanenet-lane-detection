#!/bin/sh

path2=/media/remus/datasets/AVMSnapshots/test_models/duplicate_loss_flip/inference_models
input_graph=graph20.pb
input_checkpoint=model20.ckpt
output_graph=frozen7.pb
output_node_names=lanenet_model/inference/decode/pix_embedding_relu,binary_seg

#path=/media/ionut/storage1/experiments/tensorflow/tensorflow/python/tools
path=/home/remusm/tf16/lib/python3.6/site-packages/tensorflow/python/tools
python $path/freeze_graph.py --input_graph $path2/$input_graph --input_checkpoint $path2/$input_checkpoint --output_graph $path2/$output_graph --output_node_names $output_node_names 