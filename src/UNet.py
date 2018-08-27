from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import datetime
import sys, os
from glob import glob
import imageio
import scipy.misc as misc
import matplotlib.pyplot as plt
sys.path.append(os.path.realpath('./src/data_io'))

import TensorflowUtils as utils
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
    tf.flags.DEFINE_string("logs_dir", "UNet_VOC_logs/", "path to logs directory")
    tf.flags.DEFINE_string("test_dir", "data/test/", "path of test image")
    tf.flags.DEFINE_float("learning_rate", "5e-5", "Learning rate for Adam Optimizer")
    tf.flags.DEFINE_float("decay_rate", "0.5", "Learning rate for Adam Optimizer")
    tf.flags.DEFINE_float("dropout_rate", "0.3", "dropout rate")
    tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
    tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")
    return FLAGS


class Unet:
    
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
        self.image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="input_image")
        self.annotation = tf.placeholder(tf.int32, shape=[None, image_size, image_size, num_class], name="annotation")
        self.phase_train = tf.placeholder(tf.bool, name='phase_train')
        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        
        # Prediction and loss
        self.pred_annotation, self.image_segment_logits = \
            self.inference(self.image, self.keep_probability, self.phase_train, self.flags)
        image_segment = tf.nn.softmax(self.image_segment_logits)
        colorized_annotation = tf.argmax(self.annotation, axis=3)
        colorized_annotation = tf.expand_dims(colorized_annotation, dim=3)
        self.colorized_annotation = utils.batch_colorize(
                                    colorized_annotation, 0, num_class, self.flags.cmap)
        self.colorized_pred_annotation = utils.batch_colorize(
                                            self.pred_annotation, 0, num_class, self.flags.cmap)
        self.loss = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.image_segment_logits, labels=self.annotation, name="entropy")))
        
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
        tf.summary.image("annotation", self.colorized_annotation, max_outputs=2)
        tf.summary.image("pred_annotation", self.colorized_pred_annotation, max_outputs=2)
        self.loss_summary = tf.summary.scalar("total_loss", self.loss)
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
        
        lr = self.sess.run(self.learning_rate)
        for itr in range(self.flags.max_iteration):
            if itr != 0 and itr % self.flags.decay_steps == 0:
                lr *= self.flags.decay_rate
                self.sess.run(tf.assign(self.learning_rate, lr))
            
            train_images, train_annotations = train_dataset_reader.next_batch(self.flags.batch_size)
            feed_dict = {self.image: train_images, self.annotation:train_annotations,
                            self.keep_probability: self.flags.dropout_rate, self.phase_train: True}
            
            self.sess.run(self.train_op, feed_dict=feed_dict)
            
            if itr % 10 == 0:
                train_loss, summary_str = self.sess.run([self.loss, self.loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                self.train_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                valid_images, valid_annotations = validation_dataset_reader.get_random_batch(self.flags.batch_size)
                valid_loss, summary_sva = self.sess.run([self.loss, self.summary_op], 
                    feed_dict={self.image: valid_images, self.annotation: valid_annotations, 
                                self.keep_probability: 1.0, self.phase_train: False})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                self.validation_writer.add_summary(summary_sva, itr)
                self.saver.save(self.sess, os.path.join(self.flags.logs_dir, "model.ckpt"), itr)
        return
    
    def visaulize_pred(self, dataset_reader):
        """
        Predict segmentation of images random selected from dataset_reader.
        """
        
        valid_images, valid_annotations = dataset_reader.get_random_batch(self.flags.batch_size)
        feed_dict = {self.image: valid_images, self.keep_probability: 1.0, self.phase_train: False}
        
        pred = self.sess.run([self.pred_annotation], feed_dict=feed_dict)
        pred = np.squeeze(pred, axis=3)
        pred = utils.batch_colorize_ndarray(pred, 
                                0, self.flags.num_class, self.flags.cmap)[:,:,:,:3]
        valid_annotations = np.argmax(valid_annotations, axis=-1)
        valid_annotations = utils.batch_colorize_ndarray(valid_annotations, 
                                0, self.flags.num_class, self.flags.cmap)[:,:,:,:3]
        
        for itr in range(self.flags.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), self.flags.logs_dir, name="inp_" + str(5+itr),mean=1)
            utils.save_image(valid_annotations[itr].astype(np.uint8), self.flags.logs_dir, name="gt_" + str(5+itr),mean=1)
            utils.save_image(pred[itr].astype(np.uint8), self.flags.logs_dir, name="pred_" + str(5+itr),mean=1)
            print("Saved image: %d" % itr)
        
        return valid_images, pred
    
    def predict_segmentation(self, images):
        feed_dict = {self.image: images, self.keep_probability: 1.0, self.phase_train: False}
        pred = self.sess.run([self.pred_annotation], feed_dict=feed_dict)
        pred = np.squeeze(pred, axis=3)
        return pred
    
    def plot_segmentation_under_test_dir(self):
        
        image_pattern = os.path.join(self.flags.test_dir, '*')
        image_lst = glob(image_pattern)
        data = []
        if not image_lst:
            print('No files found')    
        else:
            test_images = np.stack([
                    misc.imresize(imageio.imread(file), 
                    [self.flags.image_size, self.flags.image_size], 
                    interp='bilinear')
                for file in image_lst])        
            test_preds = self.predict_segmentation(test_images)
            colorized_test_preds = utils.batch_colorize_ndarray(test_preds, 
                                    0, self.flags.num_class, self.flags.cmap)[:,:,:,:3]
            for i, (imag, pred) in enumerate(zip(test_images, colorized_test_preds)):
                fig, axes = plt.subplots(1,2)
                axes[0].imshow(imag)
                axes[1].imshow(pred)
                axes[0].axis('off')
                axes[1].axis('off')
                filename = os.path.join(self.flags.logs_dir, 'Figure_%d.png'%i)
                plt.savefig(filename, dpi=300, format="png", transparent=False)
            # plt.show()
        return test_images, test_preds
    
    @classmethod
    def unet_encode(cls, image, keep_prob, phase_train, index_module):
        """
        Each encoder module double the number of channels,
        2 conv2x2, relu, droupout, batchnorm
        """
        
        in_ch = image.get_shape()[-1].value
        
        if index_module == 0:
            out_ch = 64
        else:
            out_ch = in_ch * 2
            
        # Convolution
        name = 'conv_%d'%index_module
        with slim.arg_scope([slim.conv2d], padding='SAME',
                weights_initializer=tf.keras.initializers.he_normal()):
            conv1 = slim.conv2d(image,out_ch,3,activation_fn=None,scope=name+'_1')
            conv2 = slim.conv2d(conv1,out_ch,3,activation_fn=tf.nn.relu,scope=name+'_2')
        relu_dropout = slim.dropout(conv2, keep_prob=keep_prob)
        # relu_dropout_bn = slim.batch_norm(relu_dropout, is_training=phase_train)
        return relu_dropout
    
    @classmethod
    def unet_decode(cls, image, keep_prob, phase_train, index_module):
        """
        Each encoder module keep the same number of channels,
        2 conv3x3, relu, droupout, batchnorm.
        """
        in_ch = image.get_shape()[-1].value
        out_ch = in_ch // 2
        
        # Convolution
        name = 'conv_%d'%index_module
        with slim.arg_scope([slim.conv2d], padding='SAME',
                weights_initializer=tf.keras.initializers.he_normal()):
            conv1 = slim.conv2d(image,out_ch,3,activation_fn=None,scope=name+'_1')
            conv2 = slim.conv2d(conv1,out_ch,3,activation_fn=tf.nn.relu,scope=name+'_2')
        relu_dropout = slim.dropout(conv2, keep_prob=keep_prob)
        # relu_dropout_bn = slim.batch_norm(relu_dropout, is_training=phase_train)
        return relu_dropout
        
    @classmethod
    def upconv(cls, image, index_module):
        """
        upconvolute by a 2x2 kernel.
        """
        in_ch = image.get_shape()[-1].value
        out_ch = in_ch // 2
        name = 'upconv%d'%index_module
        upconv = slim.conv2d_transpose(image,out_ch,2,stride=2,
                                    weights_initializer=tf.keras.initializers.he_normal(),
                                    padding='SAME',activation_fn=None,scope=name)
        return upconv
    
    @classmethod
    def unet(cls, image, keep_prob, phase_train, output_channel, num_layers, is_debug=False):
        
        net = {}
        batch_size = tf.shape(image)[0]
        current = image
        net['image'] = current
        for index_module in range(num_layers):
            # Check type of module
            is_encoder = index_module < num_layers//2
            is_decoder = index_module > num_layers//2
            is_classifier = index_module == num_layers//2
            
            # Set number of input and output channels
            in_ch = current.get_shape()[-1]
            mod_output = 'mod%d_out'
            if is_encoder:
                current = cls.unet_encode(current, keep_prob, phase_train, index_module)
                name = mod_output%index_module
                net[name] = current
                current = slim.max_pool2d(current, [2, 2], stride = 2, padding='SAME')
                
            if is_classifier:
                current = cls.unet_encode(current, keep_prob, phase_train, index_module)
                name = mod_output%index_module
                net[name] = current
                current = cls.upconv(current, index_module)
                
            if is_decoder:
                fuse_pool = mod_output%(num_layers-1-index_module)
                # print(index_module, num_layers-1-index_module)
                # print(net[fuse_pool].get_shape())
                # print(current.get_shape())
                current = tf.concat([current, net[fuse_pool]], axis=3, name="fuse_%d"%index_module)
                current = cls.unet_decode(current, keep_prob, phase_train, index_module)
                name = mod_output%index_module
                net[name] = current
                if index_module != num_layers-1:
                    current = cls.upconv(current, index_module)
            if is_debug:
                print(name)
                print(net[name].get_shape())
                utils.add_activation_summary(current)
        
        # conv1x1
        current = slim.conv2d(current, output_channel, 1)
        name = 'segment'
        net[name] = current
        if is_debug:
            print(name)
            print(net[name].get_shape())
            print('unet complete')
        return net
    
    @classmethod
    def inference(cls, image, keep_prob, phase_train, flags):
        """
        Semantic segmentation by UNet.
        
        Args:
            image: input image. Should have values in range 0-255
            keep_prob: keep_probability of dropout layers
            phase_train: training phase for batch normalization layers [true / false] 
                        (currently is disable)
            flags: tf_flags
        return:
            pred_annotation: argmax of prediction
            image_segment_logits: predicted segmentation without softmax.
        """
        
        num_layers = flags.num_layers
        with tf.variable_scope("inference"):
            encoder_net = cls.unet(image, keep_prob, phase_train, 
                            flags.num_class, num_layers, flags.debug)
        image_segment_logits = encoder_net['segment']
        softmax_image_segment = tf.nn.softmax(image_segment_logits,name='softmax_logits')
        pred_annotation = tf.argmax(softmax_image_segment, axis=3, name="prediction")
        pred_annotation = tf.expand_dims(pred_annotation, axis=3)
        return pred_annotation, image_segment_logits
    
    @classmethod
    def train(cls, loss_val, var_list, flags):
        """
        Create train_op and learning_rate.
        """

        learning_rate = tf.Variable(flags.learning_rate, trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        if flags.debug:
            # print(len(var_list))
            for grad, var in grads:
                utils.add_gradient_summary(grad, var)
        train_op = optimizer.apply_gradients(grads)
        return learning_rate, train_op


if __name__ == '__main__':
    """
    Init network and train. 
    """
    
    flags = tf_flags()
    net = Unet(flags)
    
    print("Setting up dataset reader")
    train_dataset_reader, validation_dataset_reader = create_BatchDatset()
    
    if flags.mode == "train":
        net.train_net(train_dataset_reader, validation_dataset_reader)
        
    elif flags.mode == "visualize":
        valid_images, preds = net.visaulize_pred(validation_dataset_reader)
        
    elif flags.mode == "test":
        test_images, test_preds = net.plot_segmentation_under_test_dir()
