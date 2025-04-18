# alzheimersdetection

A project to detect Alzheimers using MRI scan images.
I have used a dataset consisting of 6400 images to classify the scan as Mild Demented,Moderate Demented,Non Demented, and Very Mild Demented.
It has a test accuracy of 96%

first preprocess the dataset to split it into train,test and val
then train the model with the datasplit with atleast 20 epoch nums
then test the model for accuracy
you can test the model with individual images also using the predict.py code where we use our image path.
