"""
Modified(heavily) from TensorFlow unet implementation.
This file contains the Trainer class which takes a modified-unet model and trains it.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging
import tensorflow as tf
import sys

class Trainer(object):
    """
    Trains a unet instance
    
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    
    """
    
    def __init__(self, net, batch_size=1, norm_grads=False, optimizer="momentum", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs
        
    def _get_optimizer(self, training_iters, global_step):
        if self.optimizer == "momentum":
            learning_rate = self.opt_kwargs.pop("learning_rate", 0.2)
            decay_rate = self.opt_kwargs.pop("decay_rate", 0.95)
            momentum = self.opt_kwargs.pop("momentum", 0.2)
            
            self.learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=decay_rate, 
                                                        staircase=True)
            
            optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs).minimize(self.net.total_cost, 
                                                                                global_step=global_step)
        elif self.optimizer == "adam":
            learning_rate = self.opt_kwargs.pop("learning_rate",5e-04)
            self.learning_rate_node = tf.Variable(learning_rate)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node, 
                                               **self.opt_kwargs).minimize(self.net.total_cost,
                                                                     global_step=global_step)
        
        return optimizer
        
    def _initialize(self, training_iters, output_path, restore, prediction_path):
        global_step = tf.Variable(0)

        # Gradient nodes for summaries
        self.total_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.total_gradients_node)]))
        self.net.summary_list.append(tf.summary.histogram('totalgrads', self.total_gradients_node))

        # Summaries
        self.localiloss_summary = tf.Summary()
        self.classiloss_summary = tf.Summary()
        self.totalloss_summary = tf.Summary()
        self.acc10_summary = tf.Summary()
        self.acc15_summary = tf.Summary()
        self.acc20_summary = tf.Summary()
        #self.learning_rate_summary = tf.summary.scalar('learning_rate', self.learning_rate_node)
        self.locali_rmse_summary= tf.Summary()
        self.classi_accuracy_summary = tf.Summary()
        self.optimizer = self._get_optimizer(training_iters, global_step)
        

        #self.summary_op = tf.summary.merge(self.net.summary_list)        
        init = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()

        output_path = os.path.abspath(output_path)
        print(output_path)
        
        if not restore:
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)
        
        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)
            logging.info("Allocating '{:}'".format(output_path + "/TrainSummary"))
            os.makedirs(output_path + "/TrainSummary")
            logging.info("Allocating '{:}'".format(output_path + "/ValSummary"))
            os.makedirs(output_path + "/ValSummary")
        
        return init, init2

    def train(self, data_provider, output_path, training_iters=10, epochs=100, dropout=0.50, display_step = 100, restore=False, write_graph=False, prediction_path = 'prediction'):
        """
        Lauches the training process
        
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored 
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.cpkt")
        if epochs == 0:
            return save_path
        
        init, init2 = self._initialize(training_iters, output_path, restore, prediction_path)
        no_validation_batches = data_provider.no_validation_batches(10)
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)
            
            sess.run(init)
            sess.run(init2)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)
            
            train_summary_writer = tf.summary.FileWriter(output_path + "/TrainSummary", graph=sess.graph)
            
            empty1 = tf.summary.FileWriter(output_path + "/empty1", graph=sess.graph)
            self.locali_rmse_summary.value.add(tag="RMSE", simple_value = 1515)
            empty1.add_summary(self.localiloss_summary, 1)
            
            empty2 = tf.summary.FileWriter(output_path + "/empty2", graph=sess.graph)
            empty2.add_summary(self.localiloss_summary, 1)
            
            val_summary_writer = tf.summary.FileWriter(output_path + "/ValSummary", graph=sess.graph)
            logging.info("Start optimization")
            avg_gradients = None
            for epoch in range(epochs):
                tr_epoch_total_loss = 0
                tr_epoch_classi_loss = 0
                tr_epoch_locali_loss = 0
                tr_epoch_classi_acc = 0
                tr_total_pred10 = 0
                tr_total_pred15 = 0
                tr_total_pred20 = 0
                tr_total_locali_error = 0
                for step in range((epoch*training_iters), ((epoch+1)*training_iters)):
                    # Get next training batch from data_provider class
                    batch_x, batch_classi_y, batch_locali_y = data_provider.get_training_batch(self.batch_size)
                    # Run optimization op (backprop)
                    _, total_loss, classi_loss, locali_loss, lr, total_gradients, simple_entropy, locali_error = sess.run((self.optimizer,  
                                                                             self.net.total_cost,
                                                                             self.net.classi_cost, self.net.locali_cost,
                                                                             self.learning_rate_node,
                                                                             self.net.total_gradients_node, 
                                                                             self.net.simple_entropy, self.net.locali_error), 
                                                                             feed_dict={self.net.x: batch_x,
                                                                                        self.net.locali_y: batch_locali_y,
                                                                                        self.net.classi_y: batch_classi_y,
                                                                                        self.net.keep_prob: dropout})
                                                                                        
                    # Gradients for visualizing purposes                                                
                    avg_gradients = _update_avg_gradients(avg_gradients, total_gradients, step)
                    norm_gradients = [np.linalg.norm(gradient) for gradient in avg_gradients]
                    self.total_gradients_node.assign(norm_gradients).eval()
                    #self.classi_gradients_node.assign(classi_gradients).eval()
                    #self.locali_gradients_node.assign(locali_gradients).eval()

                    if step % display_step == 0:
                        # Record minibatch stats in tensorflow for the training minibatch
                        pred10, pred15, pred20, classi_acc = self.output_minibatch_stats(epoch, display_step, sess, train_summary_writer, step, batch_x, batch_classi_y, batch_locali_y, tag="TRAINING")

                        # Record minibatch stats in tensorflow for the validation minibatch
                        val_batch_x, val_batch_classi_y, val_batch_locali_y = data_provider.get_validation_batch(10)
                        self.output_minibatch_stats(epoch, display_step, sess, val_summary_writer, step, val_batch_x, val_batch_classi_y, val_batch_locali_y, tag="VALIDATION")
                        del val_batch_x
                        del val_batch_classi_y
                        del val_batch_locali_y
                    
                    tr_epoch_total_loss += total_loss
                    tr_epoch_classi_loss +=  classi_loss
                    tr_epoch_locali_loss += locali_loss
                    tr_epoch_classi_acc += classi_acc
                    tr_total_pred10 += pred10
                    tr_total_pred15 += pred15
                    tr_total_pred20 += pred20
                    tr_total_locali_error += np.mean(locali_error)

                    del batch_x
                    del batch_classi_y
                    del batch_locali_y

                # Print on console the per-epoch stats of the trainig set
                self.output_epoch_stats(epoch, tr_total_pred10, tr_total_pred15, tr_total_pred20, tr_epoch_total_loss, tr_epoch_classi_loss, tr_epoch_locali_loss, tr_epoch_classi_acc, training_iters, tr_total_locali_error, lr)

                
                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")
            
            return save_path

    def output_validation_stats(self, data_provider, sess, val_summary_writer, epoch, step, graph = True):
        # This function is not being used anywhere currently. Was being used in previous incarnations of the models.
        # Left here intentionally for probable future use. Don't bother reading through it.
 
        batchsize=2
        batches = data_provider.no_validation_batches(batchsize)
        print("batches are:" + str(batches))
        val_total_loss = 0
        val_classi_loss = 0
        val_locali_loss = 0
        val_locali_error = 0
        #val_classi_acc = 0
        """
        val_locali_error = np.zeros((10,))
        
        total_pred10 = np.zeros((10,))
        total_pred15 = np.zeros((10,))
        total_pred20 = np.zeros((10,))
        """
        for i in range(batches):
            val_x, val_classi_y, val_locali_y = data_provider.get_validation_batch(batchsize)
            total_loss, classi_loss, locali_loss, locali_error, classi_accuracy, lr = sess.run([self.net.total_cost,
                                                                self.net.classi_cost,
                                                                self.net.locali_cost, 
                                                                self.net.locali_error,
                                                                self.net.classi_accuracy,
                                                                self.learning_rate_node], 
                                                                feed_dict={self.net.x: val_x,
                                                                           self.net.locali_y: val_locali_y,
                                                                           self.net.classi_y : val_classi_y,
                                                                           self.net.keep_prob: 1.})
            """
            pred10 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y: val_locali_y,
                                                                    self.net.classi_y : val_classi_y,
                                                                    self.net.threshold:10,
                                                                    self.net.keep_prob: 1.})

            pred15 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y:val_locali_y,
                                                                    self.net.classi_y : val_classi_y,
                                                                    self.net.threshold:15,
                                                                    self.net.keep_prob: 1.})

            pred20 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: val_x,
                                                                    self.net.locali_y: val_locali_y,
                                                                    self.net.classi_y : val_classi_y,
                                                                    self.net.threshold:20,
                                                                    self.net.keep_prob: 1.})
            
            #import pdb; pdb.set_trace()
 
            """
            val_total_loss += total_loss
            val_classi_loss += classi_loss
            val_locali_loss += locali_loss
            #val_classi_acc += classi_accuracy
            val_locali_error += np.mean(locali_error)
            #total_pred10 += pred10
            #otal_pred15 += pred15
            #total_pred20 += pred20

            del val_x
            del val_classi_y
            del val_locali_y
        val_total_loss /= float(batches)
        val_classi_loss /= float(batches)
        val_locali_loss /= float(batches)
        #val_classi_acc /= float(batches)
        val_locali_error /=float(batches)
        #total_pred10 /=float(batches)
        #total_pred15 /= float(batches)
        #total_pred20 /= float(batches)
        # Write loss summary

        #import pdb; pdb.set_trace()
        # Write accuracy summary
        #summary = tf.Summary(value=[tf.Summary.Value(tag="val_accuracy", simple_value=total_acc),])
        #val_summary_writer.add_summary(summary, epoch)
        #val_summary_writer.flush()
        if graph:
            self.localiloss_summary.value.add(tag="locali_loss", simple_value = val_locali_loss)
            self.classiloss_summary.value.add(tag="classi_loss", simple_value = val_classi_loss)
            self.totalloss_summary.value.add(tag="total_loss", simple_value = val_total_loss)
            self.locali_rmse_summary.value.add(tag="RMSE", simple_value = val_locali_error)
            #self.acc10_summary.value.add(tag="acc10", tensor = tf.make_tensor_proto(total_pred10))
            #self.acc15_summary.value.add(tag="acc15", tensor = tf.make_tensor_proto(total_pred15))
            #self.acc20_summary.value.add(tag="acc20", tensor = tf.make_tensor_proto(total_pred20))
            """
            for i in range(len(self.acc10_summary)):
                tag_name =  "acc10_" + "joint_" + str(i)
                self.acc10_summary[i].value.add(tag=tag_name, simple_value = total_pred10[i])
            for i in range(len(self.acc10_summary)):
                tag_name =  "acc15_" + "joint_" + str(i)
                self.acc15_summary[i].value.add(tag=tag_name, simple_value = total_pred15[i])
            for i in range(len(self.acc10_summary)):
                tag_name =  "acc20_" + "joint_" + str(i)
                self.acc20_summary[i].value.add(tag=tag_name, simple_value = total_pred20[i])
            
            self.classi_accuracy_summary.value.add(tag="classi_acc", simple_value = val_classi_acc)
            """
            #
            """
            for i in range(len(self.acc10_summary)):
                tag_name =  "locali_rmse_" + "joint_" + str(i)
                self.locali_rmse_summary[i].value.add(tag=tag_name, simple_value = val_locali_error[i])
            """
            val_summary_writer.add_summary(self.localiloss_summary, epoch)
            val_summary_writer.add_summary(self.classiloss_summary, epoch)
            val_summary_writer.add_summary(self.totalloss_summary, epoch)
            val_summary_writer.add_summary(self.locali_rmse_summary,epoch)
            """
            for i in range(len(self.acc10_summary)):
                val_summary_writer.add_summary(self.acc10_summary[i], step) 
                val_summary_writer.add_summary(self.acc15_summary[i], step)
                val_summary_writer.add_summary(self.acc20_summary[i], step)
                val_summary_writer.add_summary(self.locali_rmse_summary[i],step)
            
            val_summary_writer.add_summary(self.classi_accuracy_summary, step)
            #val_summary_writer.add_summary(self.locali_rmse_summary, step)
            """
            val_summary_writer.flush()


        """      
        logging.info("Validation: epoch= {:},step = {:}, Total Loss= {:.8f}, classi_loss= {:.8f}, locali_loss= {:.8f}, classi_accuracy= {:.8f}, accuracy10= {:.8f}, \
                     accuracy15= {:.8f}, accuracy20= {:.8f}, RMSE error= {:.8f}, learning_rate={:.8f} \n".format(epoch, step,
                                                                                        val_total_loss,
                                                                                        val_classi_loss,
                                                                                        val_locali_loss,
                                                                                        val_classi_acc,
                                                                                        np.mean(total_pred10),
                                                                                        np.mean(total_pred15),
                                                                                        np.mean(total_pred20),
                                                                                        float(np.mean(val_locali_error)),
                                                                                        lr))
        """
        logging.info("Validation: Epoch = {:}, step = {:}, val_total_loss= {:.8f}, val_classi_loss= {:.8f} ,val_locali_loss= {:.8f} ,val_locali_error={:.8f}, learning rate: {:.8f} \n".format((epoch),step, float(val_total_loss),float(val_classi_loss),
                      float(val_locali_loss ), float(val_locali_error),float(lr)))

    def output_epoch_stats(self, epoch, total_pred10, total_pred15, total_pred20, epoch_total_loss, epoch_classi_loss, epoch_locali_loss, epoch_classi_acc, training_iters, total_locali_error, lr):
        # Output on console the epoch stats of the training batch
        logging.info("Training: Epoch = {:},  Average pred10={:.8f}, Average pred15={:.8f}, Average pred20={:.8f}, epoch_total_loss= {:.8f},epoch_classi_loss= {:.8f} ,epoch_locali_loss= {:.8f} \
                      , epoch_classi_acc = {:.8f},locali_error={:.8f}, learning rate: {:.8f} \n".format(epoch,  float(total_pred10 / training_iters), 
                      float(total_pred15 / training_iters), float(total_pred20 / training_iters), float(epoch_total_loss / training_iters),float(epoch_classi_loss / training_iters),
                      float(epoch_locali_loss /training_iters), float(epoch_classi_acc / training_iters), float(total_locali_error / training_iters),float(lr)))
    
    def output_minibatch_stats(self, epoch, display_step, sess, summary_writer, step, batch_x, batch_classi_y, batch_locali_y, tag="TRAINING"):
        # Calculate batch loss and accuracy
        # Write training set summary
        total_loss, classi_loss, locali_loss, locali_error, classi_accuracy, lr = sess.run([self.net.total_cost,
                                                            self.net.classi_cost,
                                                            self.net.locali_cost, 
                                                            self.net.locali_error,
                                                            self.net.classi_accuracy,
                                                            self.learning_rate_node], 
                                                            feed_dict={self.net.x: batch_x,
                                                                       self.net.locali_y: batch_locali_y,
                                                                       self.net.classi_y : batch_classi_y,
                                                                       self.net.keep_prob: 1.})

        pred10 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.classi_y : batch_classi_y,
                                                                self.net.threshold:10,
                                                                self.net.keep_prob: 1.})

        pred15 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.classi_y : batch_classi_y,
                                                                self.net.threshold:15,
                                                                self.net.keep_prob: 1.})

        pred20 = sess.run(self.net.locali_accuracy, feed_dict={self.net.x: batch_x,
                                                                self.net.locali_y: batch_locali_y,
                                                                self.net.classi_y : batch_classi_y,
                                                                self.net.threshold:20,
                                                                self.net.keep_prob: 1.})
        pred10 = np.mean(pred10)
        pred15 = np.mean(pred15)
        pred20 = np.mean(pred20)

        locali_error = np.mean(locali_error)
        self.localiloss_summary.value.add(tag="locali_loss", simple_value = locali_loss)
        self.classiloss_summary.value.add(tag="classi_loss", simple_value = classi_loss)
        self.totalloss_summary.value.add(tag="total_loss", simple_value = total_loss)
        self.acc10_summary.value.add(tag="acc10", simple_value = pred10)
        self.acc15_summary.value.add(tag="acc15", simple_value = pred15)
        self.acc20_summary.value.add(tag="acc20", simple_value = pred20)
        self.classi_accuracy_summary.value.add(tag="classi_acc", simple_value = classi_accuracy)
        self.locali_rmse_summary.value.add(tag="RMSE", simple_value = locali_error)

        summary_writer.add_summary(self.localiloss_summary, step)
        summary_writer.add_summary(self.classiloss_summary, step)
        summary_writer.add_summary(self.totalloss_summary, step)
        summary_writer.add_summary(self.acc10_summary, step) 
        summary_writer.add_summary(self.acc15_summary, step)
        summary_writer.add_summary(self.acc20_summary, step)
        summary_writer.add_summary(self.locali_rmse_summary,step)
        summary_writer.add_summary(self.classi_accuracy_summary, step)
        summary_writer.flush()

        logging.info(tag + " TRAINING: Iter {:}, Minibatch: Total Loss= {:.8f}, classi_loss= {:.8f}, locali_loss= {:.8f}, classi_accuracy= {:.8f}, accuracy10= {:.8f}, \
                accuracy15= {:.8f}, accuracy20= {:.8f}, Minibatch RMSE error= {:.8f}, learning_rate={:.8f} \n".format(step,
                                                                                        total_loss,
                                                                                        classi_loss,
                                                                                        locali_loss,
                                                                                        classi_accuracy,
                                                                                        pred10,
                                                                                        pred15,
                                                                                        pred20,
                                                                                        float(locali_error),
                                                                                        lr))
        return pred10, pred15, pred20, classi_accuracy

def _update_avg_gradients(avg_gradients, gradients, step):
    if avg_gradients is None:
        avg_gradients = [np.zeros_like(gradient) for gradient in gradients]
    for i in range(len(gradients)):
        avg_gradients[i] = (avg_gradients[i] * (1.0 - (1.0 / (step+1)))) + (gradients[i] / (step+1))
        
    return avg_gradients


