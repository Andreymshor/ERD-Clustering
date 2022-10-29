'''A helper function to update config for pretrained model'''
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import cv2
#config = config_util.get_configs_from_pipeline_file('/home/andrusha/Desktop/Projects/ERD-Clustering/pre_trained_models/efficientdet_d6_coco17_tpu-32/pipeline.config')
#print(config)
labels = [{'name':'entity', 'id':1}, {'name':'weak_entity', 'id':2}, {'name':'rel', 'id':3}, {'name':'ident_rel', 'id':4}, {'name':'rel_attr', 'id':5}, {'name':'many', 'id':6}, {'name':'one', 'id':7}]

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile('/home/andrusha/Desktop/Projects/ERD-Clustering/my_models/my_ssd_mobnet/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config', "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = len(labels)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = '/home/andrusha/Desktop/Projects/ERD-Clustering/pre_trained_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0'
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= '/home/andrusha/Desktop/Projects/ERD-Clustering/ERDs/label_map.pbtxt'
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ['/home/andrusha/Desktop/Projects/ERD-Clustering/label_records/train.record']
pipeline_config.eval_input_reader[0].label_map_path = '/home/andrusha/Desktop/Projects/ERD-Clustering/ERDs/label_map.pbtxt'
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ['/home/andrusha/Desktop/Projects/ERD-Clustering/label_records/test.record']

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile('/home/andrusha/Desktop/Projects/ERD-Clustering/my_models/my_ssd_mobnet/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config', "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)

# command to run for training: 
'''
python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=/home/andrusha/Desktop/Projects/ERD-Clustering/pre_trained_models/efficientdet_d6_coco17_tpu-32 --pipeline_config_path=/home/andrusha/Desktop/Projects/ERD-Clustering/pre_trained_models/efficientdet_d6_coco17_tpu-32/pipeline.config --num_train_steps=2000

python Tensorflow/models/research/object_detection/model_main_tf2.py --model_dir=/home/andrusha/Desktop/Projects/ERD-Clustering/my_models/my_ssd_mobnet/ --pipeline_config_path=/home/andrusha/Desktop/Projects/ERD-Clustering/pre_trained_models/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/pipeline.config --num_train_steps=2000
'''
