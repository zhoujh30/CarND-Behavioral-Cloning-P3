# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used what I had learned about deep neural networks and convolutional neural networks to clone driving behavior. I trained, validated and tested a model using [Keras](https://keras.io/). The model output a steering angle to an autonomous vehicle.

The goals / steps of this project are the following:
* Use the [simulator](https://github.com/udacity/self-driving-car-sim) to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[image1]: ./images/img.png "Original Image"
[image2]: ./images/img_flip.png "Image Flipped"
[image3]: ./images/img_crop.png "Image Cropped"
[image4]: ./images/loss.png "Loss"
[image5]: ./images/center_img.png "Center Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.h5) containing a trained convolution neural network
* [readme.md](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/README.md) summarizing the results
* [video.mp4](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/video.mp4) recording of the vehicle driving autonomously around the track using trained network


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy


#### 1. An appropriate model architecture has been employed

I used [Nvidia CNN architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) used to train self-driving cars ([model.py line 103-119](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py#L103-L119)). 

Here is a summary of the architecture:

```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================

lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1, 33, 64)     0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           dropout_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          2459532     flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================

Total params: 2,712,951
Trainable params: 2,712,951
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting ([model.py line 112](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py#L112)). 

By splitting the data sets into 80% for training and 20% for validation, the model was trained and validated on different data sets to ensure that the model was not overfitting ([model.py line 98-99](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py#L98-L99)). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually ([model.py line 124](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/model.py#L124)).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. To capture good driving behavior, I recorded ten laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

To augment the data set, I used the images from left and right cameras by adding 0.2 correction on the steering angles:

![alt text][image1]

I also flipped images and angles:

![alt text][image2]

After the collection process, I had 7,980 data points. I then preprocessed this data by cropping the images to only keep the portions that contain useful information.

![alt text][image3]


I finally randomly shuffled the data set and put 20% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. 


### Results


Here is a visualization of the training and validation loss for each epoch in the final round of training:

![alt text][image4]

After several rounds of trainings and parameter tuning, the vehicle is able to drive autonomously around the track without leaving the road.

<p align="center">
  <img src="https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/images/video.gif?raw=true">
</p>

![video](https://github.com/zhoujh30/CarND-Behavioral-Cloning-P3/blob/master/images/video.gif?raw=true)