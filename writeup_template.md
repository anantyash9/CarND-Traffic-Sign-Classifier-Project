# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[image1]: ./examples/output_6_0.png "Visualization"
[image2]: ./examples/output_7_1.png "Grayscaling"
[image3]: ./examples/output_7_3.png "Grayscaling"
[image4]: ./examples/output_7_5.png "Grayscaling"
[image5]: ./examples/output_10_1.png "Grayscaling"
[image6]: ./examples/output_10_2.png "Grayscaling"
[image7]: ./examples/output_10_4.png "Grayscaling"
[image8]: ./examples/output_10_5.png "Grayscaling"
[image9]: ./examples/output_10_5.png "Grayscaling"
[image10]: ./examples/output_10_7.png "Grayscaling"
[image11]: ./examples/output_10_8.png "Grayscaling"
[image12]: ./examples/output_10_10.png "Grayscaling"
[image13]: ./examples/output_10_11.png "Grayscaling"
[image14]: ./examples/output_10_13.png "Grayscaling"
[image15]: ./examples/output_10_14.png "Grayscaling"
[image16]: ./examples/output_19_1.png "Grayscaling"
[image17]: ./examples/output_19_2.png "Grayscaling"
[image18]: ./examples/output_25_0.png "Grayscaling"
[image19]: ./examples/output_25_1.png "Grayscaling"
[image20]: ./examples/output_25_2.png "Grayscaling"
[image21]: ./examples/output_25_3.png "Grayscaling"
[image22]: ./examples/output_25_4.png "Grayscaling"
[image23]: ./web_image/1_big.jpg "Grayscaling"
[image24]: ./web_image/11_big.jpg "Grayscaling"
[image26]: ./web_image/24_big.jpg "Grayscaling"
[image27]: ./web_image/25_big.jpg "Grayscaling"
[image28]: ./web_image/39_big.jpg "Grayscaling"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:


* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630 
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 
It is a bar chart showing how the data is distributed between the various classes and the number of images for each class in traning, validation and test dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color is not really required to identify a traffic sign.

Here is an example of a traffic sign image before and after grayscaling.

After Greyscaling I noticed that some of the examples were too dark to see and greyscaling them was making them worse. 
I used OpenCV's CLAHE (Contrast Limited Adaptive Histogram Equalization) function after greyscaling to improve the overall contrast.


![alt text][image2]



![alt text][image3]

The test images are easier to identify after preprocessing. The edges and shape of signs are more pronounced. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							|
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x30 	|
| RELU					|												|
| Max pooling	      	| Kernal 2x2 ,stride - 2 ,outputs 13x13x30 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 9x9x60 	|
| RELU					|												|
| Max pooling	      	| Kernal 2x2 ,stride - 2 ,outputs 4x4x60 				|
|Flatten|           									|
| Dropout		|       Keep prob 0.7  								|
| Fully connected 				| input 960 , output 43       									|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the following hyper parameter

* Batch size - 512 : I was traning on AWS Instance which had sufficient memory I increased the batch size.
* Epoch - 70 : I Increased the Epoch from 30 to 50 and then to 70 as the model kept on improving. I ran it for 90 epochs but it showed no Improvement so I stuck with 70.
* Learning Rate - 0.001 : The Learning rate that was used in LeNet was 0.001. Making any change to this always resulted in bad accuracy so I left it as is.
* Optimizer - Adam Optimizer : I tried ADADGRAD Optimizer as well but Adam Optimizer outperformed it in terms of accuracy of the model.
* Mean 0 & Variance 0.1 : These were used for making the initial values of weights make a beautiful gaussian distribution.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.967 
* test set accuracy of 0.944

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  The First architecture i used was the LeNet without any changes. I chose it because it was a good starting off point to experiment with hyper parameters and adding/removing layers.
* What were some problems with the initial architecture?
  The initial model didn't improve much when running for more than 30 epocs. It was much faster to train but the accuracy was stuck below 90%. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I was getting low accuracy on both traning and validation. So I had to make the model more complex but instead of adding more layers I decided to make the convolution layers deeper. The first convolution layer in my model outputes 26x26x30 whereas the first convolution layer in LeNet's output had a depth of only 6. Similaryl the second convolution bumps the depth to 60.

* Which parameters were tuned? How were they adjusted and why?
Almost all prarmeters were tuned. See point 3 for the reasons.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

I added dropout just after flattening the convolution layer with a keep probablity of 0.7. The idea behind this is that the second copnvolution layer would look for features in the image like circles and squares which can be used by the fully connected layer to classify the sign.So, adding the dropout will force the network to learn to associate more features to a traffic sign and not rely on just a few.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image23] ![alt text][image24] ![alt text][image26] 
![alt text][image27] ![alt text][image28]

The first image might be difficult to classify because of the background of buildings and trees
In the second image a small part of another sign bellow it is visible which migh make it difficult to classify.
The third image is of a sign has some texture on the sign itself and some cars in the background.
The fourth one does have a clear background but the sign is not as clear as the others.
The fifth sign again has textures/reflections on the sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit 30km/hr     		| Speed Limit 30km/hr								| 
| Right-of-way at next intersection    			| Right-of-way at next intersection   					|
| Road Narrows on Right			| General Caution											|
| Road work      		| Road work					 				|
| Keep Left			| Keep Left	     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is much lower than the accuracy on the test set of 94.4%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




