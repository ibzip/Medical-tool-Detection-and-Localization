import tables
import numpy as np
import os

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
            llabels = self._Testllabels[self.indices_test[self.test_offset:],:]
            self.test_offset += batch_size
            
        if self.test_offset >= len(self.indices_test):
            print("Test epoch happening after this return")
            # An epoch happened on test data
            # shuffle TEST data indices
            np.random.shuffle(self.indices_test)
            self.test_offset = 0
        return images, clabels, llabels

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

    def get_training_batch(self, n):
        return self.batch_handler.get_next_training_batch(n)

    def get_validation_batch(self, n):
        return self.batch_handler.get_next_validation_batch(n)

    def get_test_batch(self, n):
        return self.batch_handler.get_next_test_batch(n)

if __name__ == "__main__":
    print("Damnnnnnnnnnnnnnnnn sonnnnnnnnnnnnnnn")

