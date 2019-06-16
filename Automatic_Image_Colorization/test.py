#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test model."""

__author__ = 'Chong Guo <armourcy@email.com>'
__copyright__ = 'Copyright 2018, Chong Guo'
__license__ = 'GPL'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from config import batch_size, display_step, test_saving_step, summary_path, testing_summary
from common import init_model
from image_helper import concat_images


if __name__ == '__main__':
    # Init model
    is_training, _, _, loss, predict_rgb, color_image_rgb, gray_image, file_paths, names = init_model(train=False)
    print("File path length is " + str(len(file_paths)))
    # Init scaffold, hooks and config
    saving_step=test_saving_step

    scaffold = tf.train.Scaffold()
    checkpoint_hook = tf.train.CheckpointSaverHook(checkpoint_dir=summary_path, save_steps=saving_step, scaffold=scaffold)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, gpu_options=(tf.GPUOptions(allow_growth=True)))
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold, config=config, checkpoint_dir=summary_path)

    # Create a session for running operations in the Graph
    with tf.train.MonitoredSession(session_creator=session_creator, hooks=[checkpoint_hook]) as sess:
        print("🤖 Start testing...")
        step = 0
        avg_loss = 0

        while not sess.should_stop():
            step += 1

            l, pred, color, gray, images = sess.run([loss, predict_rgb, color_image_rgb, gray_image, names], feed_dict={is_training: False})
            # Print batch loss
            print("📖 Testing iter %d, Minibatch Loss = %f" % (step, l))
            avg_loss += l
            # Save all testing image
            print(" The number of actual images is " + str(len(color)))
            for i in range(len(color)):
                #summary_image = concat_images(gray[i], pred[i])
                #summary_image = concat_images(summary_image, color[i])
		summary_image = pred[i]
                #plt.imsave("%s/images/%d_%d.png" % (testing_summary, step, i), summary_image)
                plt.imsave("%s/images/%s" % (testing_summary, file_paths[i][file_paths[i].rfind('/')+1:]), summary_image)
            if step >= len(file_paths) / batch_size:
                break

        print("🎉 Testing finished!")
        print("👀 Total average loss: %f" % (avg_loss / len(file_paths)))
