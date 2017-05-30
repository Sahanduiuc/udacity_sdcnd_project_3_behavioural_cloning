# Behavioral Cloning

### Introduction

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Rubric Points

I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

[//]: # (Image References)

[image1]: ./examples/center_lane.png "Center Lane Driving"
[image2]: ./examples/augmented_image.png "Augmented Image"
[image3]: ./examples/sample_angle_distribution.png "Angle Distribution"
[image4]: ./examples/epoch_history.png "Epoch History"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `drive.py` for driving the car in autonomous mode
* `model.py` contains the script to create and train the model
* `model.h5` contains a trained convolution neural network 
* `writeup_report.md` summarises the results
* `run1_20_mph.mp4`
* `run2_30_mph.mp4`
* various `notebooks` for exploration of the data and the modelling process

### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with:

```sh
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_5 (Cropping2D)        (None, 80, 320, 3)    0           cropping2d_input_5[0][0]         
____________________________________________________________________________________________________
averagepooling2d_5 (AveragePooli (None, 40, 160, 3)    0           cropping2d_5[0][0]               
____________________________________________________________________________________________________
lambda_5 (Lambda)                (None, 40, 160, 3)    0           averagepooling2d_5[0][0]         
____________________________________________________________________________________________________
conv_1 (Convolution2D)           (None, 18, 78, 6)     456         lambda_5[0][0]                   
____________________________________________________________________________________________________
conv_2 (Convolution2D)           (None, 7, 37, 12)     1812        conv_1[0][0]                     
____________________________________________________________________________________________________
conv_3 (Convolution2D)           (None, 2, 17, 48)     14448       conv_2[0][0]                     
____________________________________________________________________________________________________
spatialdropout2d_7 (SpatialDropo (None, 2, 17, 48)     0           conv_3[0][0]                     
____________________________________________________________________________________________________
flatten (Flatten)                (None, 1632)          0           spatialdropout2d_7[0][0]         
____________________________________________________________________________________________________
fc_1 (Dense)                     (None, 100)           163300      flatten[0][0]                    
____________________________________________________________________________________________________
fc_2 (Dense)                     (None, 50)            5050        fc_1[0][0]                       
____________________________________________________________________________________________________
fc_3 (Dense)                     (None, 10)            510         fc_2[0][0]                       
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             11          fc_3[0][0]                       
====================================================================================================
Total params: 185,587
Trainable params: 185,587
Non-trainable params: 0
____________________________________________________________________________________________________


```

`model.py` lines 103-149 define the model architecture.

The model includes <span style="background-color: #FFFF00">RELU layers to introduce nonlinearity (lines 125, 127, 129),</span> and the data is <span style="background-color: #FFFF00">normalized in the model using a Keras lambda layer (code line 122).</span>

### 2. Attempts to reduce overfitting in the model

The model contains <span style="background-color: #FFFF00">`SpatialDropout2D`layers in order to reduce overfitting (model.py lines 133).</span>

Additional data was gathered to help generalise, and a generator with random functions in it augmented data to further provide extra samples.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually <span style="background-color: #FFFF00">(model.py line 147).</span>

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving on track 1 and track 2, as well as reverse direction around track 1 and track 2. This data isn't "needed" to get a working model however, I included it after getting my first working model to see if it would respond better to sharp corners (it did), and due to the increased sample size I added the SpatialDropout2D layer then.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the nVidia architecture. I thought this model might be appropriate because it was domain specific.

I had issues getting a wroking model, this came down to error in my generator function, variable scope
 
 * I had a variable `lines` instead of `sample` in the generator, however earlier in the notebook I was using I had the last line in memory from reading the csv file. I also had the incorrect index.
 * However the model actually got to work with this flawed data, as it was effectively only seeing 3 angles, -25, 0, and +25, so the network architecture was able to interpolate for all cases in between it would seem.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I adjusted the speed in `drive.py` and noticed the relationship between `correction angle` and speed being driven. To improve the driving behavior in these cases, I included the extra sim data I had collected, instead of using purely the augmented sample data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. It is a bit wobbly on straights when @ 30mph, however fairly stable driving is seen @ 20mph.

### 2. Final Model Architecture

This has already been detailed earlier.

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving in both directions. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle driving around track 2 in both directions, as well.

To augment the data sat, I also flipped images and angles thinking that this would provide additional samples. Images were also randomly brightened. Center camera images were excluded when the steering angle was too small also. Below is an example of a left camera image that has been dimmed slightly

![alt text][image2]


After the collection process, I had 17,303 number of data points. 

```sh
source                                   n    total        %
====================================     =====   ======   ======

sample: total                            8,036             46.4%
----------------------------------------------------------------

sim track 1: direction=default           2,719
sim track 1: direction=reverse           2,568
sim track 1: total                                5,287    30.6%
----------------------------------------------------------------

sim track 2: direction=default           2,568
sim track 2: direction=reverse           1,412
sim track 2: total                                3,980    23.0%
----------------------------------------------------------------

totals (x3 for centre + left + right)            17,303   100.0%
================================================================
```

I then augmented this data by putting it through a generator to take advantage of Keras' parallelism. This gave the following angle distribution:

![alt text][image3]

Preprocessing was also done in Keras, resize > crop > normalise.

A parameter was passed to the `fit_generator` method to split the train/validation data up.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the following

![alt text][image4]

I used an adam optimizer so that manually training the learning rate wasn't necessary.
