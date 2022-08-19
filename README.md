# Introduction

Currently, the first part of my master's thesis is displayed in this repository, and part-time it is homework for the course "Vision and Perception" in 
The Sapienza University of Rome.

# I Part

In the first part of this project, I built a binary classifier for fire and none images. Initially, I planned to create a multi-classier for the images of fire, smoke and none, however, after several attempts, I have to admit that the smoke classification results were terrible. The baselines metrics for the smoke data were poor, and the heat map showed that the network was looking at the background, not at the target. This could be because the data was synthetic. 

The binary classifier has oustanding results and has high accuracy and recall for both classes. For this classifier, I used a pre-trained ResNet18. I have a chose it after I tried ResNet34 and ResNet50. I used all the layers except the last one. The last layer was changed and trained on my data. To reduce overfitting, I used a learning rate scheduler that changes the learning rate after 4th and 10th epochs. I must addmit after first epochs I encountered a slight underfitting, but later the model stabilized.

Finally, I studied my model with the help of various metrics and graphs. You can find more explicit findings in my notebook.

# Files

Folder Structure **Fire Classifier**:

* Early_fire_detection_classifier.ipynb -Notebook with EDA and modeling parts
* functions.py - A Py file with visualization and repeating functions. The main goal is to reduce the size of the notebook
* requirements.txt - Requirements for functions.py
* Model folder - contains the classifier model file
