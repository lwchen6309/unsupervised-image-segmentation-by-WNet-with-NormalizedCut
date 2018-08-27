from __future__ import print_function
import tensorflow as tf
import numpy as np
import datetime
import sys, os
sys.path.append(os.path.realpath('./src/data_io'))

import TensorflowUtils as utils
from WNet_naive import Wnet_naive
from soft_ncut import soft_ncut, brightness_weight, gaussian_neighbor, convert_to_batchTensor
from data_io.BatchDatsetReader_VOC import create_BatchDatset


def tf_flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
    tf.flags.DEFINE_integer("image_size", "128", "batch size for training")
    tf.flags.DEFINE_integer('max_iteration', "50000", "max iterations")
    tf.flags.DEFINE_integer('decay_steps', "10000", "max iterations")
    tf.flags.DEFINE_integer('num_class', "21", "number of classes for segmentation")
    tf.flags.DEFINE_integer('num_layers', "5", "number of layers of UNet")
    tf.flags.DEFINE_string("cmap", "viridis", "color map for segmentation")
    tf.flags.DEFINE_string("logs_dir", "WNet_bright_VOC_logs/", "path to logs directory")
    tf.flags.DEFINE_string("test_dir", "data/test/", "path of test image")
    tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
    tf.flags.DEFINE_float("decay_rate", "0.5", "Decay rate of learning_rate")
    tf.flags.DEFINE_float("dropout_rate", "0.65", "dropout rate")
    tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
    tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
    return FLAGS


class Wnet_bright(Wnet_naive):    
    
    def __init__(self, flags):
        """
        Initialize:
            placeholder,
            train_op,
            summary,
            session,
            saver and file_writer
        """
        
        self.flags = flags
        image_size = int(self.flags.image_size)
        num_class = int(self.flags.num_class)
        
        # Place holder
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="input_image")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.neighbor_indeces = tf.placeholder(tf.int64, name="neighbor_indeces")
        self.neighbor_vals = tf.placeholder(tf.float32, name="neighbor_vals")
        self.neighbor_shape = tf.placeholder(tf.int64, name="neighbor_shape")
        neighbor_filter = (self.neighbor_indeces, self.neighbor_vals, self.neighbor_shape)
        _image_weights = brightness_weight(self.image, neighbor_filter, sigma_I = 0.05)
        image_weights = convert_to_batchTensor(*_image_weights)
        
        # Prediction and loss
        self.pred_annotation, self.image_segment_logits, self.reconstruct_image = \
            self.inference(self.image, self.keep_probability, self.phase_train, self.flags)
        image_segment = tf.nn.softmax(self.image_segment_logits)
        self.colorized_pred_annotation = utils.batch_colorize(
                                    self.pred_annotation, 0, num_class, self.flags.cmap)
        self.reconstruct_loss = tf.reduce_mean(tf.reshape(
                                    (self.image - self.reconstruct_image)**2, shape=[-1]))        
        batch_soft_ncut = soft_ncut(self.image, image_segment, image_weights)
        self.soft_ncut = tf.reduce_mean(batch_soft_ncut)
        self.loss = self.reconstruct_loss + self.soft_ncut
        
        # Train var and op
        trainable_var = tf.trainable_variables()
        encode_trainable_var = tf.trainable_variables("infer_encode")
        if self.flags.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)
        self.reconst_learning_rate, self.train_reconst_op = \
            self.train(self.reconstruct_loss, trainable_var, self.flags)
        self.softNcut_learning_rate, self.train_softNcut_op = \
            self.train(self.soft_ncut, encode_trainable_var, self.flags)    
        self.reconst_learning_rate_summary = tf.summary.scalar("reconst_learning_rate", self.reconst_learning_rate)
        self.softNcut_learning_rate_summary = tf.summary.scalar("softNcut_learning_rate", self.softNcut_learning_rate)
        
        # Summary
        tf.summary.image("input_image", self.image, max_outputs=2)
        tf.summary.image("reconstruct_image", self.reconstruct_image, max_outputs=2)
        tf.summary.image("pred_annotation", self.colorized_pred_annotation, max_outputs=2)   
        reconstLoss_summary = tf.summary.scalar("reconstruct_loss", self.reconstruct_loss)
        softNcutLoss_summary = tf.summary.scalar("soft_ncut_loss", self.soft_ncut)
        totLoss_summary = tf.summary.scalar("total_loss", self.loss)
        self.loss_summary = tf.summary.merge([reconstLoss_summary, softNcutLoss_summary, totLoss_summary])
        self.summary_op = tf.summary.merge_all()
        
        # Session ,saver, and writer
        print("Setting up Session and Saver...")
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=2)
        # create two summary writers to show training loss and validation loss in the same graph
        # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
        self.train_writer = tf.summary.FileWriter(os.path.join(self.flags.logs_dir, 'train'), self.sess.graph)
        self.validation_writer = tf.summary.FileWriter(os.path.join(self.flags.logs_dir, 'validation'))
        
        print("Initialize tf variables")
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(self.flags.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...") 
        return
        
    def train_net(self, train_dataset_reader, validation_dataset_reader):
        
        image_shape = self.image.get_shape().as_list()[1:3]
        gauss_indeces, gauss_vals = gaussian_neighbor(image_shape, sigma_X = 4, r = 5)
        weight_shapes = np.prod(image_shape).astype(np.int64)
        
        # self.sess.run(tf.assign(self.reconst_learning_rate, self.flags.learning_rate))
        # self.sess.run(tf.assign(self.softNcut_learning_rate, self.flags.learning_rate))
        reconst_lr = self.sess.run(self.reconst_learning_rate)
        softNcut_lr = self.sess.run(self.softNcut_learning_rate)
        
        for itr in range(self.flags.max_iteration):
            if itr != 0 and itr % self.flags.decay_steps == 0:
                reconst_lr *= self.flags.decay_rate
                softNcut_lr *= self.flags.decay_rate
                self.sess.run(tf.assign(self.reconst_learning_rate, reconst_lr))
                self.sess.run(tf.assign(self.softNcut_learning_rate, softNcut_lr))
            
            train_images, _ = train_dataset_reader.next_batch(self.flags.batch_size)
            feed_dict = {self.image: train_images, 
                        self.keep_probability: self.flags.dropout_rate, 
                        self.phase_train: True,
                        self.neighbor_indeces: gauss_indeces,
                        self.neighbor_vals: gauss_vals,
                        self.neighbor_shape: [weight_shapes, weight_shapes]}
            valid_feed_dict = dict(feed_dict)
            
            self.sess.run(self.train_reconst_op, feed_dict=feed_dict)
            self.sess.run(self.train_softNcut_op, feed_dict=feed_dict)
            
            if itr % 100 == 0:
                train_loss, summary_str = self.sess.run([self.loss, self.loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                self.train_writer.add_summary(summary_str, itr)
            
            if itr % 1000 == 0:
                valid_images, _ = validation_dataset_reader.get_random_batch(self.flags.batch_size)
                valid_feed_dict[self.image] = valid_images
                valid_feed_dict[self.keep_probability] = 1.0
                valid_feed_dict[self.phase_train] = False
                valid_loss, summary_sva = self.sess.run([self.loss, self.summary_op], 
                    feed_dict = valid_feed_dict)
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                self.validation_writer.add_summary(summary_sva, itr)
                self.saver.save(self.sess, os.path.join(self.flags.logs_dir, "model.ckpt"), itr)
        return

if __name__ == '__main__':
    """
    Init network and train. 
    """
    
    flags = tf_flags()
    net = Wnet_bright(flags)
    
    if flags.mode == "test":
        test_images, preds = net.plot_segmentation_under_test_dir()
    
    else:
        print("Setting up dataset reader")
        train_dataset_reader, validation_dataset_reader = create_BatchDatset()
        
        if flags.mode == "train":
            net.train_net(train_dataset_reader, validation_dataset_reader)
            
        elif flags.mode == "visualize":
            valid_images, preds = net.visaulize_pred(validation_dataset_reader)
        