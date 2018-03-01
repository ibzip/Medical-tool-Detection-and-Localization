# Medical-tool-Detection-and-Localization
In this project a recent method for detection and localization of tools in minimally-invasive surgeries is implemented.
This project implements a modified-unet based model for the detection of tools and localization of positions of joints of 
these tools. The methodology has been listed in the paper present in this repository.

To train the network:
1. Clone the repository.
2. Execute DataPreparation.py script to produce the training and test-set HDF5 files.
3. Run the project_unet.py to train the model.

To test the network:
1. Use either of the test scripts provided in the repo.

