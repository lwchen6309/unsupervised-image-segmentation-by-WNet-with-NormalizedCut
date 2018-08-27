from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import datetime
import sys, os
sys.path.append(os.path.realpath('./src/data_io'))

import TensorflowUtils as utils
from data_io.BatchDatsetReader_VOC import create_BatchDatset
from UNet import Unet


def tf_flags():
    FLAGS = tf.flags.FLAGS
    tf.flags.DEFINE_integer("batch_size", "5", "batch size for training")
    tf.flags.DEFINE_integer("image_size", "128", "batch size for training")
    tf.flags.DEFINE_integer('max_iteration', "50000", "max iterations")
    tf.flags.DEFINE_integer('decay_steps', "10000", "max iterations")
    tf.flags.DEFINE_integer('num_class', "21", "number of classes for segmentation")
    tf.flags.DEFINE_integer('num_layers', "5", "number of layers of UNet")
    tf.flags.DEFINE_string("cmap", "viridis", "color map for segmentation")
    tf.flags.DEFINE_string("logs_dir", "WNet_naive_VOC_logs/", "path to logs directory")
    tf.flags.DEFINE_string("test_dir", "data/test/", "path of test image")
    tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
    tf.flags.DEFINE_float("decay_rate", "0.5", "Decay rate of learning_rate")
    tf.flags.DEFINE_float("dropout_rate", "0.65", "dropout rate")
    tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
    tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
    return FLAGS


class Wnet_naive(Unet):
    
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
        self.pred_annotation, self.image_segment_logits, self.reconstruct_image = \
            self.inference(self.image, self.keep_probability, self.phase_train, self.flags)
        self.colorized_pred_annotation = utils.batch_colorize(self.pred_annotation, 0, num_class, self.flags.cmap)
        self.loss = tf.reduce_mean(tf.reshape((self.image - self.reconstruct_image)**2, shape=[-1]))
        
        # Train var and op
        trainable_var = tf.trainable_variables()
        if self.flags.debug:
            for var in trainable_var:
                utils.add_to_regularization_and_summary(var)
        self.learning_rate, self.train_op = self.train(self.loss, trainable_var, self.flags)
        self.learning_rate_summary = tf.summary.scalar("learning_rate", self.learning_rate)
        
        # Summary
        print("Setting up summary op...")
        tf.summary.image("input_image", self.image, max_outputs=2)
        tf.summary.image("reconstruct_image", self.reconstruct_image, max_outputs=2)
        tf.summary.image("pred_annotation", self.colorized_pred_annotation, max_outputs=2)    
        self.loss_summary = tf.summary.scalar("total_loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        
        # Session ,saver, and writer
        print("Setting up Session and Saver...")
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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
        
        lr = self.sess.run(self.learning_rate)
        for itr in range(self.flags.max_iteration):
            
            if itr != 0 and itr % self.flags.decay_steps == 0:
                lr *= self.flags.decay_rate
                self.sess.run(tf.assign(self.learning_rate, lr))
            
            train_images, train_annotations = train_dataset_reader.next_batch(self.flags.batch_size)
            feed_dict = {self.image: train_images, self.keep_probability: self.flags.dropout_rate, 
                        self.phase_train: True}
            self.sess.run(self.train_op, feed_dict=feed_dict)
            
            if itr % 10 == 0:
                train_loss, summary_str = self.sess.run([self.loss, self.loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                self.train_writer.add_summary(summary_str, itr)
            
            if itr % 100 == 0:
                valid_images, _ = validation_dataset_reader.get_random_batch(self.flags.batch_size)
                valid_loss, summary_sva = self.sess.run([self.loss, self.summary_op], 
                    feed_dict={self.image: valid_images, self.keep_probability: 1.0, 
                                self.phase_train: False})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                
                # add validation loss to TensorBoard
                self.validation_writer.add_summary(summary_sva, itr)
                self.saver.save(self.sess, os.path.join(self.flags.logs_dir, "model.ckpt"), itr)
        return

    def visaulize_pred(self, dataset_reader):
        """
        Predict segmentation of images random selected from dataset_reader.
        """
        
        valid_images, _ = dataset_reader.get_random_batch(self.flags.batch_size)
        
        feed_dict = {self.image: valid_images, self.keep_probability: 1.0, self.phase_train:False}
        reconst_image, pred = self.sess.run([self.reconstruct_image, self.pred_annotation], feed_dict=feed_dict)
        pred = np.squeeze(pred, axis=3)
        pred = utils.batch_colorize_ndarray(pred, 0, self.flags.num_class, self.flags.cmap)[:,:,:,:3]
        
        for itr in range(self.flags.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), self.flags.logs_dir, name="inp_" + str(5+itr),mean=1)
            utils.save_image(reconst_image[itr].astype(np.uint8), self.flags.logs_dir, name="gt_" + str(5+itr),mean=1)
            utils.save_image(pred[itr].astype(np.uint8), self.flags.logs_dir, name="pred_" + str(5+itr),mean=1)
            print("Saved image: %d" % itr)
        
        return valid_images, pred

    def predict_segmentation(self, images):
        feed_dict = {self.image: images, self.keep_probability: 1.0, self.phase_train:False}
        reconst_image, pred = self.sess.run([self.reconstruct_image, self.pred_annotation], feed_dict=feed_dict)
        pred = np.squeeze(pred, axis=3)
        return pred
        
    @classmethod
    def inference(cls, image, keep_prob, phase_train, flags):
        """
        Semantic segmentation network definition.
        Two Unet is stack as an autoencoder.
        
        Args:
            image: input image. Should have values in range 0-255
            keep_prob: keep_probability of dropout layers
            phase_train: training phase for batch normalization layers [true / false] 
                        (currently is disable)
            flags: tf_flags
        return:
            pred_annotation: argmax of prediction
            image_segment_logits: predicted segmentation without softmax.
            reconstruct_image.
        """
        
        with tf.variable_scope("infer_encode"):
            encoder_net = cls.unet(image, keep_prob, phase_train, 
                            flags.num_class, flags.num_layers, flags.debug)
        image_segment_logits = encoder_net['segment']
        softmax_image_segment = tf.nn.softmax(image_segment_logits,name='softmax_logits')
        pred_annotation = tf.argmax(softmax_image_segment, axis=3, name="prediction")
        pred_annotation = tf.expand_dims(pred_annotation, axis=3)
        with tf.variable_scope("infer_decode"):
            decoder_net = cls.unet(softmax_image_segment, keep_prob, phase_train,
                                3, flags.num_layers, flags.debug)
        reconstruct_image = decoder_net['segment']
        
        return pred_annotation, image_segment_logits, reconstruct_image
    
    
if __name__ == '__main__':
    """
    Init network and train. 
    """
    
    flags = tf_flags()
    net = Wnet_naive(flags)

    if flags.mode == "test":
        test_images, preds = net.plot_segmentation_under_test_dir()
    
    else:
        print("Setting up dataset reader")
        train_dataset_reader, validation_dataset_reader = create_BatchDatset()
        
        if flags.mode == "train":
            net.train_net(train_dataset_reader, validation_dataset_reader)
            
        elif flags.mode == "visualize":
            valid_images, preds = net.visaulize_pred(validation_dataset_reader)
        
