"""
Modified(heavily) from TensorFlow unet implementation.
This file implements the unet class which creates a model with:
  - Downsampling layers at the end of which is a fc classification layer with sigmoid activations equal
    to the max number of tools(this implementation assumes 2) that may be present in the images.
  - At the end of the downsampling layers, it builds equivalent number of up-sampling layers(Joint localization path). Up-convolutions
    (or deconvolutions so to say) are used in building the upsampling path. The upsampling path outputs 10
    224*224 maps of probabilities, equaling 5 maps for 5 joints of 1 tool. So this model is based on assumption
    that there are at max two tools present in the images and each have 5 joints that we need to predict. The number of maps at
    the end of the localization path can be modified easily. Number of classification sigmoids can be altered as well. But this
    model is pretty much hard-coded for 2 tools and 5 joints each. Bad code, I know. Too lazy to change it. That's what you get for a
    semester project from a procrastinator.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from tf_unet_modified.layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat,
                            sigmoid_likelihood, localization_prediction)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = 224
    ny = 224
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    # down layers
    # One convolution only for 1 layer
    features = 0
    for layer in range(0, layers): # we want an extra convoluton of 2048 at the end of contraction path
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        #w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        #b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)

        # Apply RELU now
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        dw_h_convs[layer] = tmp_h_conv
        #conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        #dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        print(dw_h_convs[layer])
        
        weights.append(w1)
        biases.append(b1)
        convs.append(conv1)
        
        size -= 4
        if layer < layers - 1:
            print("IN")
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
    print("here")
    
    # do a convolution of 128 feature maps + RELU
    in_node = dw_h_convs[layers-1]
    print("final conv")
    print(in_node) # layer of 2048
    prev_features = features
    features = 128
    stddev = np.sqrt(2 / (filter_size**2 * features))
    w1 = weight_variable([filter_size, filter_size, prev_features, features], stddev)
    b1 = bias_variable([features])
    weights.append(w1)
    biases.append(b1)
    conv1 = conv2d(in_node, w1, keep_prob)
    convs.append(conv1)
    tmp_h_conv = tf.nn.relu(conv1 + b1)
    dw_h_convs[layers] = tmp_h_conv
    print("128")
    print(tmp_h_conv)
    # fully connected layer1 of 512
    fc1 = tf.layers.dense(tf.contrib.layers.flatten(tmp_h_conv), 512, name='fc1')

    # fully connected layer 2 of 256
    fc2 = tf.layers.dense(fc1, 256, name='fc2')

    # final fully connected logits layer of n_class
    class_logits = tf.layers.dense(fc2, n_class)

    print("logits:")
    print(class_logits)
    # Append a dense layer above at the end of contracting path above.
    # Now make the expansion path
    
    in_node = dw_h_convs[layers-1]
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        b1 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        in_node = tf.nn.relu(conv1 + b1)
        up_h_convs[layer] = in_node

        weights.append(w1)
        biases.append(b1)
        convs.append(conv1)
        
        size *= 2
        size -= 4
    

    # Output Map
    weight = weight_variable([1, 1, features_root, 10], stddev)# 10 = 2_tools*5_joints_per_tool
    bias = bias_variable([10])
    conv = conv2d(in_node, weight, tf.constant(1.0))
    convs.append(conv)
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map
    
    summary_list = []
    if summaries:
        
        summary_list.append(tf.summary.image('output_feature_map', get_image_summary(tf.nn.softmax(output_map))) )     
        for i, (c1) in enumerate(convs):
            summary_list.append(tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1)))
            
        for k in pools.keys():
            summary_list.append(tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k])))
        
        for k in deconv.keys():
            summary_list.append(tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k])))
        
        # Add activation histograms for the contraction path 
        for k in dw_h_convs.keys():
            summary_list.append(tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k]))
        # Add activation histograms for the expansion path
        for k in up_h_convs.keys():
            summary_list.append(tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k]))
   
    variables = []
    for w1 in weights:
        variables.append(w1)
        #variables.append(w2)
        
    for b1 in biases:
        variables.append(b1)
        #variables.append(b2)

    return variables, class_logits, output_map, int(in_size - size), summary_list


class Unet(object):
    """
    A unet implementation
    
    :param channels: (optional) number of channels in the input image
    :param n_class: (optional) number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """
    
    def __init__(self, channels=3, n_class=2, cost="cross_entropy", lambda_c = 1,lambda_l = 1, cost_kwargs={}, **kwargs):
        tf.reset_default_graph()
        
        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        self.threshold = tf.constant(0)
        self.x = tf.placeholder("float", shape=[None, None, None, channels]) # For images
        self.classi_y = tf.placeholder("float", shape=[None, n_class]) # For classification labels
        self.locali_y = tf.placeholder("float", shape=[None, 10, 224,224]) # For localization maps
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        self.variables, classi_logits, locali_map, self.offset, self.summary_list = create_conv_net(self.x, self.keep_prob, channels, n_class, **kwargs)
        
        self.classi_cost = self._get_cost(classi_logits, "sigmoid_cross_entropy", cost_kwargs)
        self.locali_cost = self._get_cost(locali_map, "localization_cost", cost_kwargs)

        # Hard-coded empirical weights weighing the classification loss more as localization loss was goind bonkers.
        lambda_c = 60.0/61.0
        lambda_l = 1.0/61.0

        self.total_cost = tf.add(lambda_c*self.classi_cost, lambda_l* self.locali_cost) # Loss for classification and localization combined
        
        # simply for logging purposes. Not part of the training.
        self.simple_entropy = self.get_simple_entropy(locali_map)

        # Gradient node that will be used to generate gradient histograms
        self.total_gradients_node = tf.gradients(self.total_cost,self.variables)#tf.trainable_variables())
        self.classi_gradients_node = tf.gradients(self.classi_cost,self.variables)# tf.trainable_variables())
        self.locali_gradients_node = tf.gradients(self.locali_cost, self.variables)#tf.trainable_variables())
        
        # For classification
        self.classi_predicter = sigmoid_likelihood(classi_logits)
        self.classi_accuracy = self.get_classi_accuracy(self.classi_predicter)
        
        # For localization
        self.locali_predicter_x, self.locali_predicter_y = localization_prediction(locali_map)
        self.locali_error = self.get_locali_error(self.locali_predicter_x, self.locali_predicter_y)
        self.locali_accuracy = self.get_locali_accuracy(self.locali_predicter_x, self.locali_predicter_y)

    def get_classi_accuracy(self, predicter):
        #predicted_class = tf.greater(prediction,0.5)
        correct = tf.equal(predicter, tf.equal(self.classi_y,1.0))
        accuracy = tf.reduce_mean( tf.cast(correct, 'float') )
        #correct_pred = tf.equal(self.y, predicter)
        #return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy

    def get_locali_accuracy(self, prediction_x, prediction_y):
        x = tf.cast(prediction_x,'float') #N*10
        y = tf.cast(prediction_y ,'float')#N*10
     
        temp = tf.reshape(self.locali_y,[-1, 10, 224* 224])
        index = tf.arg_max(temp,2)
        x_l = tf.cast(tf.divide(index,self.locali_y.shape[2]),'float')
        y_l = tf.cast(tf.floormod(index,self.locali_y.shape[2]), 'float')
        #x = tf.Print(x, [x[0][0], y[0][0], x[0][1], y[0][1]], "x_pr, y_pr: ")
        #x_l = tf.Print(x_l, [x_l[0][0], y_l[0][0], x_l[0][1], y_l[0][1]], "x_gr, y_gr: ")
        #x_l = tf.arg_max(self.locali_y, 2)
        #y_l = tf.arg_max(self.locali_y, 3)
        #mean(sqrt((x-x_l)^2 + (y-y_l)^2))
        final_x = tf.square(tf.subtract(x, x_l))
        final_y = tf.square(tf.subtract(y, y_l))
        final = tf.sqrt(tf.add(final_x, final_y))
        

        f = tf.less(final,tf.cast(self.threshold,'float'))
        #m = tf.Print(x, [x[0][0], y[0][0], x[0][1], y[0][1]], "x_pr, y_pr: ")
        #n = tf.Print(x_l, [x_l[0][0], y_l[0][0], x_l[0][1], y_l[0][1]], "x_gr, y_gr: ")
        return tf.reduce_mean(tf.cast(f, 'float'), axis = 0) # N*10

    def get_locali_error(self, prediction_x, prediction_y):
        x = tf.cast(prediction_x,'float') #N*10
        y = tf.cast(prediction_y ,'float')#N*10

        temp = tf.reshape(self.locali_y,[-1, 10, 224* 224])
        index = tf.arg_max(temp,2)
        x_l = tf.cast(tf.divide(index,self.locali_y.shape[2]),'float')
        y_l = tf.cast(tf.floormod(index,self.locali_y.shape[2]), 'float')

        final_x = tf.square(tf.subtract(x, x_l))
        final_y = tf.square(tf.subtract(y, y_l))
        a = tf.add(final_x, final_y)
        b = tf.sqrt(tf.cast(a, 'float'))
        final = tf.reduce_mean(b, axis = 0) #N*10
        return final
        
    def get_simple_entropy(self, logits):
        t_logits = tf.transpose(logits, [0, 3, 1, 2]) # change dims to N,10,224,224
            
        resh_logits = tf.reshape(t_logits, [-1,10, 224*224])
        resh_labels = tf.reshape(self.locali_y, [-1,10,224*224])
        abc = tf.nn.softmax_cross_entropy_with_logits(logits=resh_logits, labels=resh_labels, dim=2)
        return abc
        
    def _get_cost(self, logits, cost_name, cost_kwargs):
        """
        Constructs the cost functionss.
        Optional arguments are: 
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        if cost_name == "sigmoid_cross_entropy":
            tool_wise_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels = self.classi_y)
            sample_wise_loss = tf.reduce_sum(tool_wise_loss, 1)
            loss = tf.reduce_mean(sample_wise_loss)
            
        elif cost_name == "localization_cost":
        
            t_logits = tf.transpose(logits, [0, 3, 1, 2]) # change dims to N,12,224,224
            
            resh_logits = tf.reshape(t_logits, [-1,10, 224*224])
            resh_labels = tf.reshape(self.locali_y, [-1,10,224*224])

            # Now for each sample, we have 10 joints, and for each joint 224*224 classes
            
            softmax_logits = tf.nn.softmax(resh_logits)    # N*10*(224*224)
            cross_entropy_loss = -resh_labels*tf.log(softmax_logits) # N*10*(224*224)
            cross_entropy_loss1 = tf.reduce_sum(cross_entropy_loss,axis = 2) #N*10
            cross_entropy_loss2 = tf.reduce_sum(cross_entropy_loss1,axis = 1) # N
            loss = tf.reduce_mean(cross_entropy_loss2) # 1
            
            #loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=resh_logits, labels=resh_labels, dim=2),1))
        
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        regularizer = cost_kwargs.pop("regularizer", None)
        if regularizer is not None:
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
            loss += (regularizer * regularizers)
            
        return loss

    # This function is used for predicting the test dataset localization
    def locali_predict(self, model_path, x_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], 10,224,224))
            
            prediction = sess.run([self.locali_predicter_x, self.locali_predicter_y], feed_dict={self.x: x_test, self.locali_y: y_dummy, self.keep_prob: 1.})
            
        return prediction # prediction is N*10 x position, N*10 y positions
        
    # This function is used for predicting the test dataset classification
    def classi_predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], self.n_class))
            prediction = sess.run(self.classi_predicter, feed_dict={self.x: x_test, self.classi_y: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        model_path = os.path.join(model_path, "model.cpkt")
        print(model_path)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V

