import numpy as np 
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
from tf_unet_modified import unet
import tables


class BatchGenerator:
    def __init__(self, train_images, train_classi_labels, train_locali_labels, test_images, test_classi_labels,
                 test_locali_labels):
        self._timages = train_images
        self._tclabels = train_classi_labels
        self._tllabels = train_locali_labels
        self._testimages = test_images
        self._testclabels = test_classi_labels
        self._testllabels = test_locali_labels

        self.train_offset = 0
        self.val_offset = 0
        self.test_offset= 0
        
        self.tr_size = self._timages.shape[0]
        self.test_size = self._testimages.shape[0]
        self.val_size = int(np.floor(self.test_size * 0.3)) # Use 30% of test set as validation
        
        self.indices_train = np.arange(self.tr_size)
        self.indices_test = np.arange(self.test_size)

        np.random.shuffle(self.indices_train)
        np.random.shuffle(self.indices_test)
        
        self.indices_val = self.indices_test[0:self.val_size]
        self.indices_test = self.indices_test[self.val_size:]
        self.test_size = len(self.indices_test)

    def get_next_training_batch(self, batch_size):
        images = []
        clabels = []
        llabels = []
        if len(self.indices_train) >= batch_size:
            images = self._timages[self.indices_train[0:batch_size],:]
            clabels = self._tclabels[self.indices_train[0:batch_size],:]
            llabels = self._tllabels[self.indices_train[0:batch_size],:]
            self.indices_train = self.indices_train[batch_size:]
        else:
            images = self._timages[self.indices_train[0:],:]
            clabels = self._tclabels[self.indices_train[0:],:]
            llabels = self._tllabels[self.indices_train[0:],:]
            self.indices_train = []
            
        if len(self.indices_train) == 0:
            print("Training epoch happening after this return")
            # An epoch happened on val data
            # shuffle VALIDATION data indices
            self.indices_train = np.arange(self.tr_size)
            np.random.shuffle(self.indices_train)
        return images, clabels, llabels
    
    def get_next_validation_batch(self, batch_size):
        images = []
        clabels = []
        tlabels = []
        if (self.val_offset + batch_size) <= len(self.indices_val):
            images = self._testimages[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            clabels = self._testclabels[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            llabels = self._testllabels[self.indices_val[self.val_offset:self.val_offset+batch_size], :]
            self.val_offset += batch_size
        else:
            images = self._testimages[self.indices_val[self.val_offset:],:]
            clabels = self._testclabels[self.indices_val[self.val_offset:],:]
            llabels = self._testllabels[self.indices_val[self.val_offset:],:]
            self.val_offset += batch_size
            
        if self.val_offset >= len(self.indices_val):
            print("Validation epoch happening after this return")
            # An epoch happened on val data
            # shuffle VALIDATION data indices
            np.random.shuffle(self.indices_val)
            self.val_offset = 0
        return images, clabels, llabels
    
    def get_next_test_batch(self, batch_size):
        images = []
        clabels = []
        tlabels = []
        if (self.test_offset + batch_size) <= len(self.indices_test):
            images = self._testimages[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            clabels = self._testclabels[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            llabels = self._testllabels[self.indices_test[self.test_offset:self.test_offset+batch_size], :]
            self.test_offset += batch_size
        else:
            images = self._testimages[self.indices_test[self.test_offset:],:]
            clabels = self._testclabels[self.indices_test[self.test_offset:],:]
            llabels = self._testllabels[self.indices_test[self.test_offset:],:]
            self.test_offset += batch_size
            
        if self.test_offset >= len(self.indices_test):
            print("Test epoch happening after this return")
            # An epoch happened on test data
            # shuffle TEST data indices
            np.random.shuffle(self.indices_test)
            self.test_offset = 0
        return images, clabels, llabels
    


# In[4]:


class DataProvider:
    """
    Class which provides data related functions
    """

    def __init__(self, trainhdf5file, testhdf5file):
        trainhdf5file = os.path.abspath(trainhdf5file)
        testhdf5file = os.path.abspath(testhdf5file)

        trainhdf5file = tables.open_file(trainhdf5file, "r")
        testhdf5file = tables.open_file(testhdf5file, "r")
        print(trainhdf5file)
        print(testhdf5file)
        
        self.train_X = np.array(trainhdf5file.root.images)/255.0
        self.train_classi_Y = np.array(trainhdf5file.root.classilabels)
        self.train_locali_Y = np.array(trainhdf5file.root.localilabels)
        
        self.test_X = np.array(testhdf5file.root.images)/255.0
        self.test_classi_Y = np.array(testhdf5file.root.classilabels)
        self.test_locali_Y = np.array(testhdf5file.root.localilabels)
    
        # To make everything fast at the expense of huge RAM usage, pass these handlers as numpy arrays 
        # to BatchGenerator
        self.batch_handler = BatchGenerator(self.train_X, self.train_classi_Y, self.train_locali_Y,
                                            self.test_X, self.test_classi_Y, self.test_locali_Y)

    def get_training_batch(self, n):
        return self.batch_handler.get_next_training_batch(n)
    
    def no_validation_batches(self, batch_size):
        if len(self.batch_handler.indices_val)%batch_size == 0:
            return len(self.batch_handler.indices_val)/batch_size
        else:
            return len(self.batch_handler.indices_val)/batch_size + 1
    
    def no_training_batches(self, batch_size):
        if len(self.batch_handler.indices_train)%batch_size == 0:
            return len(self.batch_handler.indices_train)/batch_size
        else:
            return len(self.batch_handler.indices_train)/batch_size + 1
    
    def no_test_batches(self, batch_size):
        if len(self.batch_handler.indices_test)%batch_size == 0:
            return len(self.batch_handler.indices_test)/batch_size
        else:
            return len(self.batch_handler.indices_test)/batch_size + 1
            
    def get_validation_batch(self, n):
        return self.batch_handler.get_next_validation_batch(n)

    def get_test_batch(self, n):
        return self.batch_handler.get_next_test_batch(n)

print("Loading data ...................")
trainhdf5filename = "data/Tracking_Robotic_Training/Training_set/train_images_224_10_joints.hdf5"
testhdf5filename = "data/Test_set/test_images_224_10_joints.hdf5"
data_provider = DataProvider(trainhdf5filename, testhdf5filename)

print("Importing Model....................")
# Import model
net = unet.Unet(layers=6, features_root=64, channels=3, n_class=4, cost="sigmoid_cross_entropy", lambda_c = 1, lambda_l = 1)

model_path = "model_saved"
output_path = "val_prediction_result"
print("Starting Prediction for validation set.............")        
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt and ckpt.model_checkpoint_path:
        print("Restoring Model Weights .......")
        #net.restore(sess, ckpt.model_checkpoint_path)
        net.restore(sess, model_path)
    #train_summary_writer = tf.summary.FileWriter(model_path + "/TrainSummary", graph=sess.graph)
    #val_summary_writer = tf.summary.FileWriter(model_path + "/ValSummary", graph=sess.graph)
    #test_summary_writer = tf.summary.FileWriter(model_path + "/TestSummary", graph=sess.graph)
    ##########################################################################
    # Predicting Validation Accuracy :                                         #
    ##########################################################################



    batchsize = 10
    batches = data_provider.no_validation_batches(batchsize)
    val_locali_error = 0
    val_classi_acc = 0
    val_locali_acc_summary = tf.Summary()
    val_total_locali_acc = np.zeros((10,6))
    threshold = [5,10,15,20,25,30]

    val_joint0_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint0_Summary", graph=sess.graph)
    val_joint1_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint1_Summary", graph=sess.graph)
    val_joint2_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint2_Summary", graph=sess.graph)
    val_joint3_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint3_Summary", graph=sess.graph)
    val_joint4_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint4_Summary", graph=sess.graph)
    val_joint5_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint5_Summary", graph=sess.graph)
    val_joint6_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint6_Summary", graph=sess.graph)
    val_joint7_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint7_Summary", graph=sess.graph)
    val_joint8_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint8_Summary", graph=sess.graph)
    val_joint9_summary_writer = tf.summary.FileWriter(output_path + "/val_Joint9_Summary", graph=sess.graph)
    
    for i in range(batches):
        val_locali_acc  = np.zeros((10,))

        val_x, val_classi_y, val_locali_y = data_provider.get_validation_batch(batchsize)
        print("Validation Prediction : batch_no : [%d / %d ]" %(i,batches))
        locali_error, classi_accuracy= sess.run([net.locali_error,
                                                        net.classi_accuracy], 
                                                        feed_dict={net.x: val_x,
                                                                    net.locali_y: val_locali_y,
                                                                    net.classi_y : val_classi_y,
                                                                    net.keep_prob: 1.})
        for i in range(len(threshold)):
            val_locali_acc = sess.run(net.locali_accuracy, feed_dict={net.x: val_x,
                                                                    net.locali_y: val_locali_y,
                                                                    net.classi_y : val_classi_y,
                                                                    net.threshold:threshold[i],
                                                                    net.keep_prob: 1.})
            
            val_total_locali_acc[:,i] += val_locali_acc    
        
        val_classi_acc += classi_accuracy
        val_locali_error += np.mean(locali_error)      
      
    val_total_locali_acc /= float(batches)
    val_classi_acc /= float(batches)
    val_locali_error /= float(batches)

    print("Pred _ locali val :", i)
    print(val_total_locali_acc)
    
    for i in range(len(threshold)):    
        tag_name = "pred_locali_val" #+ #str(threshold[j])  
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[0,i])
        val_joint0_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[1,i])
        val_joint1_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[2,i])
        val_joint2_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[3,i])
        val_joint3_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[4,i])
        val_joint4_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[5,i])
        val_joint5_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[6,i])
        val_joint6_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[7,i])
        val_joint7_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[8,i])
        val_joint8_summary_writer.add_summary(val_locali_acc_summary, threshold[i])
        val_locali_acc_summary.value.add(tag=tag_name, simple_value = val_total_locali_acc[9,i])
        val_joint9_summary_writer.add_summary(val_locali_acc_summary, threshold[i])       
    del val_x
    del val_classi_y
    del val_locali_y

    val_joint0_summary_writer.flush()
    val_joint1_summary_writer.flush()
    val_joint2_summary_writer.flush()
    val_joint3_summary_writer.flush()
    val_joint4_summary_writer.flush()
    val_joint5_summary_writer.flush()
    val_joint6_summary_writer.flush()
    val_joint7_summary_writer.flush()
    val_joint8_summary_writer.flush()
    val_joint9_summary_writer.flush()

    logging.info("Validation Results: Val_locali_error={:.8f}, Val_classi_acc = {:.8f} \n".format(float(val_locali_error),float(val_classi_acc)))
        
