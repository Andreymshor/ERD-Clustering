# make sure your labels in image are the same as the labels in here (along w capitalization)

# entity: looks like a rectangle
# weak entity: looks like 2 rectangles on top of each other
# rel: a relationship: looks like a diamond
# ident_rel: an identifying relationship: Looks like 2 diamonds on top of each other
# rel_attr: an attribute that belongs to a relationship, looks like an oval/eclipse
# many: appears at the end of a connection, has 3 lines coming out of it. Can have a O before the 3 lines, still counts as many.
# one: appears at the end of a connection, has 1 or 2 vertical lines like so: ||. Can have a O before the 2 lines, still counts as one.
# 
import os
import tensorflow as tf
import object_detection

CUSTOM_MODEL_NAME = 'my_efficient_det_d6'
PRETRAINED_MODEL_NAME = 'efficientdet_d6_coco17_tpu-32'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d6_coco17_tpu-32.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'

