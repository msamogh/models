from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import json

import numpy as np

import tensorflow as tf
from tensorflow.contrib import predictor


def get_predictions(saved_model_path):
    predict_fn = predictor.from_saved_model(
        saved_model_path,
        signature_def_key='probabilities'
    )

    from official.resnet.cifar10_main import input_fn
    next_element = input_fn(
        False,
        data_dir='/tmp/cifar10_data',
        batch_size=128
    )[0]

    with tf.Session() as sess:
        next_element = sess.run(next_element)
        payload = {'input': str(next_element.tolist())}
        json.dump(payload[0], open('example.json', 'w'))
        predictions = predict_fn({
            'input': next_element
        })
        return predictions['output'][0]

if __name__ == '__main__':
    get_predictions('resnet_clf_tf_estimator/1522660104')
