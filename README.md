## FIRE-Machine-Learning

This project was created as part of the First-year Introduction to Research in Engineering (FIRE) program. 

The project aims to create a program/algorithm to help detect high gradient areas in an image. The long-term hope of this is project is to integrate it with the research of Dr. Michael Cullinan and Eva Natinsky, the advisors for this project, and their work on AFM image scans. The gradient detection from this project will help distinguish feature-rich areas from feature-poor areas in these images scans, allowing for a more optimized use of the electron microscope to focus on the feature-rich area. 

This project works by creating an artificial dataset by applying a simple Sobel filter to detect high gradient areas. Afterward, all the images are passed and in a UNet machine learning architecture to create a more generalized model. The end result is available in this repository. 

# 1_fire.py:
Training and Validation Dataset is created through this python file

DIRECTORY is the path of the input dataset
OUTPUT_DIRECTORY is the path of where the images formed by the algorithm will be placed  
FILE_TYPE is the file type of the images inside the directory (*.jpg or *.png)

***Instructions:***
1) find a large dataset of image and place in the same directory as **1_fire.py**
2) change parameter **DIRECTORY** to name of directory
3) run **1_fire.py** and make sure the value inside parameter **OUTPUT_DIRECTORY** is created 
4) repeat process for smaller validation dataset of images 
5) create a new directory to store entire dataset
6) put the larger dataset images into this new directory and name the images **train_images** and the masks **train_masks**
7) repeat process for smaller dataset but instead, name the directories  **val_images** and **val_masks**
8) the final structure should be a directory with four folders within it named **train_images**, **train_masks**,  **val_images** and **val_masks**, with the respective images inside

Data I used can be downloaded from https://drive.google.com/drive/folders/1Lp_1T1ESZnpBNP2-ZR9Kn5ShSd4686uM?usp=sharing

# 2_train.py:

Model is trained through this file, and stored in .pth.tar file.

BASE_DIRECTORY is the path where the data is stored for training

***Instructions Cont.:***

8) BASE_DIRECTORY is changed to whatever the name of the master directory from above.
9) Run code

After each epoch, the validation set will be tested the outputs will be saved into a folder named **saved_images**. It contain a mask with a corresponding prediction mask, noted with a prefix "pred". 

To run the model on custom images without training, simply comment out all the functions within the NUM_EPOCHS for loop except **save_predictions_as_imgs()**

UNet Architecture taken from: https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/image_segmentation/semanic_segmentation_unet
