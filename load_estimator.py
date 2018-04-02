from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf


from tensorflow.contrib import predictor

from memory_profiler import profile

import random

predict_fn = predictor.from_saved_model(
    'resnet_clf_tf_estimator/1522231372/',
    signature_def_key='probabilities'
)

from official.resnet.cifar10_main import input_fn
next_element = input_fn(
    False,
    data_dir='/tmp/cifar10_data',
    batch_size=128
)[0]
with tf.Session() as sess:
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()[0]
    # next_element = sess.run(next_element)

    predictions = predict_fn({
        'input': next_element
    })
    print(predictions['output'][0])
    print(len(predictions['output'][0]))

