import numpy as np
import os
import tables
from tables import *
import cv2
import sys
import json
from copy import deepcopy
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from utils_exp2 import *

##################################################################################################################################
#### This class Prepares the Training and Test datasets for the task of instrument detection and instrument joing localization.###
#### It performs following tasks for both the training data and the test data:
#### 1. Read the surgery videos and extract frames.
#### 2. Resize the frames to 224*224 from original size of 576*720.
#### 3. In-case of Training data set, since there are only 940 labeled frames available, augment each frame by flipping the original
####    frame horizontally, vertically, and then flipping horizontally the vertically flipped frame.
#### 4. Write the images, classification labels representing the presence of 4 tools in each image, and the localization gaussian+uniform
####    maps for each joint of each instrument present in the image, into an HDF5 file. If an instrument is present in the image, a gaussian 
####    map is generated for each visible joint of that instrument. If a joint of any instrument is not visible, a uniform map is used as label.
###################################################################################################################################
class DataPreparation:
    
    def __init__(self, input_videos_dir, input_labels_dir, output_dir, output_file_name, gen_test_imgs=False ):
    
        self.input_videos_dir = os.path.abspath(input_videos_dir)
        self.input_labels_dir = os.path.abspath(input_labels_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.dict_videos = {}
        self.output_file_name = output_file_name
        self.gen_test_imgs = gen_test_imgs

        video_no = 0
        for sub_dir in next(os.walk(self.input_videos_dir))[1]:
            label_file_list_path = []
            for file_ in os.listdir(os.path.join(self.input_videos_dir,sub_dir)):
                if file_.endswith(".avi"):
                    video_file_path = os.path.join(self.input_videos_dir,sub_dir,file_)
                    video_no = int(sub_dir[7:]) # Parent Dir of each video is appended with "Dataset" and then number.
            if gen_test_imgs:
                self.dict_videos[video_file_path] = os.path.join(self.input_labels_dir,"test" + str(video_no) + "_labels.json") 
            else:
                self.dict_videos[video_file_path] = os.path.join(self.input_labels_dir,"train" + str(video_no) + "_labels.json") 

        if not(os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)
        #for key in self.dict_videos:
        #    print(key + ": " + self.dict_videos[key])

    # Returns the no of frames with available labels
    def get_no_images(self):
        no_images = 0
        for videofilepath in self.dict_videos:
            label_file = self.dict_videos[videofilepath]
            fh = open(label_file)
            data=json.load(fh)
            for elem in data:
                if len(elem["annotations"]) > 0:
                    no_images += 1
            fh.close()
        return no_images

    def fill_label_list(self, lis, joint_dict):
        if joint_dict["class"] == "LeftClasperPoint":
            lis[0] = float(joint_dict["x"])
            lis[1] = float(joint_dict["y"])
        elif joint_dict["class"] == "RightClasperPoint":
            lis[2] = float(joint_dict["x"])
            lis[3] = float(joint_dict["y"])
        elif joint_dict["class"] == "ShaftPoint":
            lis[4] = float(joint_dict["x"])
            lis[5] = float(joint_dict["y"])
        #elif joint_dict["class"] == "TrackedPoint":
        #    lis[6] = float(joint_dict["x"])
        #    lis[7] = float(joint_dict["y"])
        elif joint_dict["class"] == "EndPoint":
            lis[6] = float(joint_dict["x"])
            lis[7] = float(joint_dict["y"])
        elif joint_dict["class"] == "HeadPoint":
            lis[8] = float(joint_dict["x"])
            lis[9] = float(joint_dict["y"])
        else:
            return lis

        if float(joint_dict["x"]) > 720.0:
            print("Label real-world x is out of range")
        if float(joint_dict["y"]) > 576.0:
            print("Label real-world y is out of range")

        if float(joint_dict["x"]) < 0:
            print("Label real-world x less than 0")
        if float(joint_dict["y"]) < 0:
            print("Label real-world y is out of range")
        
        return lis

    def append_frame_and_label(self, frame, anno, train_store, label_store, flip_hor=False):
        # Append original frame
        train_store.append(frame[None])
        #Append label
        tool1 = [-1]*10 #Right clasper instrument joints: series of x,y,x,y --> LeftClasperPointx, LeftClasperPointy, RightClasperPointx, RightClasperPointy,
                          #ShaftPointx, ShaftPointy, TrackedPointx, TrackedPointy, EndPointx, EndPointy, HeadPointx, HeadPointy
        tool2 = [-1]*10 #Left clasper instrument joints
        tool3 = [-1]*10 #Right scissor instrument joints
        tool4 = [-1]*10 #left scissor instrument joints

        is_t1 = False
        is_t2 = False
        is_t3 = False
        is_t4 = False
        # if image was flipped horizontally, tool1(rightclasper) becomes tool2(leftclaspe) and vice versa
        # Same for scissors
        for info in anno: # info= 1joint
            if not "id" in info:
                continue 
            if flip_hor:
                if info["id"] == "tool1":
                    info["id"] = "tool2"
                elif info["id"] == "tool2":
                    info["id"] = "tool1"
                elif info["id"] == "tool3":
                    info["id"] = "tool4"
                elif info["id"] == "tool4":
                    info["id"] = "tool3"

        for info in anno:   
            if not "id" in info:
                continue 
            if info["id"] == "tool1":
                tool1 = self.fill_label_list(tool1, info)
                is_t1 = True
            elif info["id"] == "tool2":
                tool2 = self.fill_label_list(tool2, info)
                is_t2 = True
            elif info["id"] == "tool3":
                tool3 = self.fill_label_list(tool3, info)
                is_t3 = True
            elif info["id"] == "tool4":
                tool4 = self.fill_label_list(tool4, info)
                is_t4 = True
        final_label_array = []

        if is_t1:
            final_label_array += [1111] + tool1 #1111 indicated the next 12 indices are data of tool1
        if is_t2:
            final_label_array += [1110] + tool2 #1110 indicated the next 12 indices are data of tool2 and so on for below
        if is_t3:
            final_label_array += [1100] + tool3
        if is_t4:
            final_label_array += [1000] + tool4
        label_store.append(final_label_array)

    def openHdfFileWrite(self):
        # Create hdf5 file
        self.hdf5_file = tables.open_file(self.output_dir +  '/' + self.output_file_name, mode='w')
        self.tempfile = tables.open_file(self.output_dir +  '/' + "tempfile", mode='w')

    def closeHdf5File(self):
        self.tempfile.close()
        self.hdf5_file.close()

    # Extracts the frames and labels and stores in a hdf5 file
    def extractFrames(self):
        
        shape = [0,224,224, 3]
        # Create hdf5 dataset in hdf5_file for training images
        train_store = self.hdf5_file.create_earray(self.hdf5_file.root, 'images', tables.UInt8Atom(), shape=shape)
        # Create hdf5 dataset in hdf5_file for training labels
        label_store = self.tempfile.create_vlarray(self.tempfile.root, 'rawlabels', tables.Float32Atom(shape=()), "labelarray")

        print("Working.. Wait 2-3 minutes. Or pick the code and parallelize it. I am sure that will be more effort so just wait :)")

        # Store Images in the file
        imcount = 1
        for videofilepath in self.dict_videos:
            print(videofilepath)
            cap = cv2.VideoCapture(videofilepath)

            label_file = self.dict_videos[videofilepath]
            fh = open(label_file)
            data=json.load(fh)
            for elem in data:
                if not cap.isOpened():
                    break

                ret, frame = cap.read()
                if ret != True:
                    break
                #if imcount == 10:
                #    return
                original_frame_width = frame.shape[1]
                original_frame_height = frame.shape[0]
                frame = cv2.resize(frame,(224,224))

                #if half_size:
                #    # Halve the image and adjust the labels accordingly
                #    print("halving")
                anno = elem["annotations"]


                if len(anno) > 0:
                    #if (len(anno) < 6) or (len(anno) > 6 and len(anno) < 12):
                    #    print("fuckthisshit")
                    #    print(len(anno))
                    #    import pdb
                    #    pdb.set_trace()
                    ##################################################
                    ####  1 .Append the real image and labels ########
                    ##################################################

                    #gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #for info0 in anno:
                        # Flip the labels along vertical axis
                        #cv2.circle(gray0,(int(info0["x"]),int(info0["y"])), 5, (0,255,0), -1)
                    #cv2.imshow('frame0',gray0)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break

                    # append original frame and label

                    self.append_frame_and_label(frame, anno, train_store, label_store)
                    if self.gen_test_imgs:
	                #.stdout.write("processed image %d" % imcount)            
	                #print("processed classi label" + str(imcount))
	                #sys.stdout.flush()
                        print("processed image: " + str(imcount))
                        imcount += 1
                        # No need to augment the test data
                        continue
                    ####################################################
                    ## 2. append horizontally flipped frame and labels##
                    ####################################################
                    hor = np.flip(frame, 1)
                    #gray = cv2.cvtColor(hor, cv2.COLOR_BGR2GRAY)
                    new_anno = deepcopy(anno)
                    # Flip joint positions accordingly
                    for info in new_anno:
                        # Flip the labels along vertical axis
                        info["x"] = original_frame_width - float(info["x"]) + 1.0
                    #    cv2.circle(gray,(int(info["x"]),int(info["y"])), 5, (0,255,0), -1)
                    self.append_frame_and_label(hor, new_anno, train_store, label_store, flip_hor=True)
                    #cv2.imshow('frame',gray)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break


                    ############################################################################
                    ## 3.a Now append vertically flipped labels and frame of the original image#
                    ############################################################################
                    ver = np.flip(frame, 0)
                    #gray2 = cv2.cvtColor(ver, cv2.COLOR_BGR2GRAY)
                    new_anno2 = deepcopy(anno)
                    # Flip joint positions accordingly
                    for info2 in new_anno2:
                        # Flip the labels along vertical axis
                        info2["y"] = original_frame_height - float(info2["y"]) + 1.0
                    #    cv2.circle(gray2,(int(info2["x"]),int(info2["y"])), 5, (0,255,0), -1)
                    self.append_frame_and_label(ver, new_anno2, train_store, label_store)
                    #cv2.imshow('frame2',gray2)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                    #######################################################################################
                    ## 3.b Now append horizontally flipped labels and frame of the vetically flipped image#
                    #######################################################################################                  
                    ver2 = np.flip(ver, 1)
                    #gray3 = cv2.cvtColor(ver2, cv2.COLOR_BGR2GRAY)
                    new_anno3 = deepcopy(new_anno2)
                    # Flip joint positions accordingly
                    for info3 in new_anno3:
                        # Flip the labels along vertical axis
                        info3["x"] = original_frame_width - float(info3["x"]) + 1.0
                    #    cv2.circle(gray3,(int(info3["x"]),int(info3["y"])), 5, (0,255,0), -1)
                    self.append_frame_and_label(ver2, new_anno3, train_store, label_store,  flip_hor=True)
                    #cv2.imshow('frame3',gray3)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                    #    break
                    #time.sleep(0.05)
	            #sys.stdout.write("processed image %d" % imcount)            
	            #print("processed classi label" + str(imcount))
	            #sys.stdout.flush()
                    print("processed image: " + str(imcount))
                    imcount += 1

            fh.close()   
            cap.release()
            cv2.destroyAllWindows()
            #break # after 1st video

    def extractClassiLabels(self):
        
        ######################################################
        ########## Write classification task labels ##########
        ######################################################
        # Assuming at max we have 4 instruments present in any image
        shape = [0,4]
        # Create hdf5 dataset for classification labels
        classi_label_store = self.hdf5_file.create_earray(self.hdf5_file.root, 'classilabels', tables.UInt8Atom(), shape=shape)
        
        # Assuming at max we have 4 tools in an image as per our dataset
        # A.1 Form classification problem labels
        self.X = self.hdf5_file.root.images
        imcount = 1
        for image_label in self.tempfile.root.rawlabels:

            hot_vector = [0,0,0,0] # We are assuming 4 kinds of instruments can be present in the image
            if 1111 in image_label:
                # Means that right clasper instrument is present
                # Change the label at hot_vector[0] to 1
                hot_vector[0] = 1
            if 1110 in image_label:
                # Means that left clasper instrument is present
                # Change the label at hot_vector[1] to 1
                hot_vector[1] = 1
            if 1100 in image_label:
                # Means that right scissor instrument is present
                # Change the label at hot_vector[2] to 1
                hot_vector[2] = 1
            if 1000 in image_label:
                # Means that left scissor instrument is present
                # Change the label at hot_vector[3] to 1
                hot_vector[3] = 1
            classi_label_store.append(np.array(hot_vector)[None])
            print("processed classi label" + str(imcount))
            imcount += 1


    def extractLocaliLabels(self):
        ######################################################
        ########## Write localization task labels ############
        ######################################################
        # Create hdf5 dataset for localization labels
        locali_label_store = self.hdf5_file.create_earray(self.hdf5_file.root, 'localilabels', tables.Float32Atom(), shape=(0,10,224,224))
        imcount = 1
        for image_label in self.tempfile.root.rawlabels:
            p_maps = p_map_generator(image_label, image_size = (224,224))
            locali_label_store.append(p_maps[None])
            print("processed locali label" + str(imcount))
            imcount += 1
        print(locali_label_store.shape)



# <----------------Test the class here---------------->
if __name__ == "__main__":
    
    train_input_videos_dir = "data/Tracking_Robotic_Training/Training"
    train_input_labels_dir = "data/EndoVisPoseAnnotation-master/train_labels"
    train_output_dir = "Train_set/"
    train_output_file_name = "train_images_224_2_tools_10_joints.hdf5"

    test_input_videos_dir = "data/Tracking_Robotic_Testing"
    test_input_labels_dir = "data/EndoVisPoseAnnotation-master/test_labels"
    test_output_dir = "Test_set/"
    test_output_file_name = "test_images_224_2_tools_10_joints.hdf5"
    
    # A. Process Training dataset
    #############################################################################
    ##Store The images and raw labels for training and test data into hdf5 files#
    #############################################################################
    # For generating train images
    dp = DataPreparation(train_input_videos_dir,train_input_labels_dir, train_output_dir, train_output_file_name, gen_test_imgs=False)
    print("------------------Preparing Training Data--------------------- ")
    dp.openHdfFileWrite()
    dp.extractFrames()
    #########################################################################################
    ##Now process these labels to produce classification labels and store them in hdf5 files#
    #########################################################################################
    dp.extractClassiLabels()
    #########################################################################################
    ##Now process these labels to produce localization maps and store them in hdf5 files#####
    #########################################################################################
    dp.extractLocaliLabels()

    dp.closeHdf5File()
    
    print("------------------Preparing Test Data---------------------")
    # For generating test images
    dp2 = DataPreparation(test_input_videos_dir,test_input_labels_dir, test_output_dir, test_output_file_name, gen_test_imgs=True)
    dp2.openHdfFileWrite()
    dp2.extractFrames()
    #########################################################################################
    ##Now process these labels to produce classification labels and store them in hdf5 files#
    #########################################################################################
    dp2.extractClassiLabels()
    #########################################################################################
    ##Now process these labels to produce localization maps and store them in hdf5 files#####
    #########################################################################################
    dp2.extractLocaliLabels()
    dp2.closeHdf5File()

#print("------------------Augmenting Data--------------------- ")
#dp.data_augmentation(output_file_name)
