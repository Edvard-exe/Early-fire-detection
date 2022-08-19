# Introduction

Currently, the first part of my master's thesis is displayed in this repository, and part-time it is homework for the course "Vision and Perception" in 
The Sapienza University of Rome.

# I Part

In the first part of this project I have built binary classifier for fire and none images. Initialy I was planing to build multi classifier for fire, smoke
and none images however after multiple attempts I must admit that smoke that was bad. The baseline metrics for smoke data were poor and heatmap showed that
network is looking to the background not to the target. It could occur because the data was synthetic. 

Binary classifier have very good results and have high accuracy and recall for both classes. For this classifier I have used pretrained ResNet18. I have 
choose after I have tried ResNet34 and ResNet50. I have uosed all layers except last. Last layer was changed and trained on my data. In order to reduce 
overfitting I have used learning rate scheduler which have change learning rate after 4 and 10 epochs. I must addmit during first epochs I faced slight
underfitting, but later model stabilized.

Lastly I have studied my model trhougt different metrics and plots. More findings you can find in my notebook

# Fiels
