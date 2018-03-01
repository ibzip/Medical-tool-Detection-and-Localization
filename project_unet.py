
# coding: utf-8

# In[1]:


from tf_unet_modified import unet
from tf_unet_modified.Trainer import Trainer
import tensorflow as tf
import os
import tables
import numpy as np
from copy import deepcopy
import cv2
import time
import matplotlib.pyplot as plt
from data_provider import DataProvider

# IMPORT DATA

# In[ ]:


#input_videos_dir = "data/Tracking_Robotic_Training/Training/"
#input_labels_dir = "data/EndoVisPoseAnnotation-master/train_labels"
#output_dir = "data/Tracking_Robotic_Training/Training_set/"
#output_file_name = "train_images_128.hdf5"


# In[ ]:


#dp = DataPreparation(input_videos_dir,input_labels_dir, output_dir)
#print("------------------Preparing Data--------------------- ")
#dp.extractFrames(output_file_name)

#To check training set
trainhdf5filename = "Train_set/train_images_224_2_tools_10_joints.hdf5"
testhdf5filename = "Test_set/test_images_224_2_tools_10_joints.hdf5"


#preparing data loading
data_provider = DataProvider(trainhdf5filename, testhdf5filename)
#print(data_provider.train_X[0])
#plt.imshow(data_provider.train_X[0])

#plt.show()

#setup & training
# Initial channels are 3, initial output features expected are 64, 2 classes and forming 3 layers 
# where each layer does 2 convolutions with RELU and one pooling

net = unet.Unet(layers=6, features_root=64, channels=3, n_class=4, cost="sigmoid_cross_entropy", lambda_c = 1, lambda_l = 1)

trainer = Trainer(net, batch_size=10, optimizer="adam", opt_kwargs={"learning_rate":5e-04})

model_path = trainer.train(data_provider, "model_saved", training_iters= 380,epochs=40, write_graph=True) #95*40 = 3800 we have 3760 images



