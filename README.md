# Introduction

Currently, the first part of my master's thesis is displayed in this repository, and part-time it is homework for the course "Vision and Perception" in 
The Sapienza University of Rome.

# I Part

In the first part of this project, I built a binary classifier for fire and none images. Initially, I planned to create a multi-classier for the images of fire, smoke and none, however, after several attempts, I have to admit that the smoke classification results were terrible. The baselines metrics for the smoke data were poor, and the heat map showed that the network was looking at the background, not at the target. This could be because the data was synthetic. 

Binary classifier have very good results and have high accuracy and recall for both classes. For this classifier I have used pretrained ResNet18. I have 
choose after I have tried ResNet34 and ResNet50. I have uosed all layers except last. Last layer was changed and trained on my data. In order to reduce 
overfitting I have used learning rate scheduler which have change learning rate after 4 and 10 epochs. I must addmit during first epochs I faced slight
underfitting, but later model stabilized.

Lastly I have studied my model trhougt different metrics and plots. More findings you can find in my notebook

# Fiels

Structure of folder **Fire Classifier**:

* Early_fire_detection_classifier.ipynb - Notebook with EDA and modeling parts
* functions.py - Py file with visualization and repeated functions. Main purpose reduce size of notebook
* requirements.txt - Requirements for functions.py
* Model folder - contains classifier model file
