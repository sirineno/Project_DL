# Investigating Human Visual Salience Prediction Using Pretrained ANNs and linear probing

Human attention is guided by both low-level visual features (e.g., contrast, edges) and high-level semantic information (e.g., faces, objects). Visual salience, which predicts where humans look first in an image, is a key topic in understanding human visual perception.

## Project Overview

This project aims to explore the capability of pretrained artificial neural networks (ANNs) to predict human visual salience as measured by eye-tracking data. By examining the alignment between the networks' internal representations and human attention patterns, we aim to gain insights into the computational mechanisms underlying visual salience prediction.

### Objectives

- **Predict Human Visual Salience:** Investigate whether pretrained ANNs can accurately predict human visual salience using eye-tracking data.
- **Align Network Representations with Human Attention:** Examine the degree to which the internal representations of ANNs align with human attention patterns through linear probing techniques.
- **Understand Visual Perception:** Contribute to the broader understanding of human visual perception by analyzing the relationship between ANN predictions and human visual salience.

### Methodology

1. **Dataset:** Use SALICON dataset, which provides salience maps derived from crowd-sourced gaze data. We will not collect new data for this project, as the existing dataset provides sufficient coverage for our analysis. link : https://salicon.net
2. **Pretrained ANN Models:** Use the pretrained resnet50 to extract features of the images.
3. **Linear Probing:** Apply linear probing techniques to assess the alignment between the networks' extracted features and human saliency maps.
4. **Analysis:** Compare the model's predictions with the actual data to evaluate the performance of the models.


### Methods and Algorithms

1. **Pretrained Neural Networks:**
   - Use pretrained ResNet to extract image features. The model is known for capturing both low- and high-level visual information.
   
2. **Linear Probing:**
   - Apply linear probing to map the network’s features to the fixation data.
   - Train a simple linear model to predict human salience patterns using neural network representations as input.
   
3. **Salience Map Comparison:**
   - Generate salience maps from neural network predictions.
   - Compare these maps to the actual data to evaluate alignment.

### Evaluation

We will perform both qualitative and quantitative analyses to evaluate the results:

1. **Quantitative Metrics:**
   - Pearson Correlation: Assess correlation between ANN-predicted salience maps and human fixation maps.
   - AUC (Area Under the Curve): Evaluate how well network predictions match observed human fixations.
   
2. **Qualitative Analysis:**
    Visualize ANN-predicted salience maps and overlay them with human fixation heatmaps.
     
### How to go through the files in Order : 
1. We explained our model structure and the training + validation execution in the "model_train_val.ipynb". However, the execution of it was made in the python scripts in the "Train_and_val" folder. We created the notebook just to explain the model.
2. We analyzed the model's performance in the "Model_performance" Folder. You'll find in this folder the validation's metrics results file that we used to analyze the model's performance.

### How to download and use the dataset : 
I you would like to download the datasets, you can get them through this link : https://salicon.net/download/
Make sure to download : 
- Images : training and validation
- Annotations: training and validation 



