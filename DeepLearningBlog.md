# Deep Learning Blog

Author:
Meng Zheng(M.Zheng-3@student.tudelft.nl) 5241693
Annie Liu(A.Liu-6@student.tudelft.nl) 5676452

This blog records the work of reproducing paper "G-TAD: Sub-Graph Localization for Temporal Action Detection", which is originally proposed by Mengmeng Xu and al.

In this blog, firstly we will introduce the original paper, provide a background information about the aim and main contribution about such paper: Meng focuses on Hyperparameter tuning and Annie on Label Augmentation. Then, G-TAD's structure and corresponding two dataset shall be addressed. After that, we will provide some details about the process of implement G-TAD and our contributions. Also the result shall be provided and discussed. Finally each member will provide their reflection.
## Introduction
G-TAD is designed and applied based on THUMOS Challenge 2014, which is a Action Recognition challenge. This challenge involves two tasks:
* Classification task: classify the action for each video from 20 labels.
* Prediction task: predict the start frame and the end frame of corresponding action label.


In this paper, author mainly focusing on the second target, which is to predict the range of such action label. As for the classification task, they applied pretrained weight from TSM: Temporal Shift Module for Efficient Video Understanding.
## Model Architecture
G-TAD is combined of three modules. Which are:
* Feature extraction module
* SGAlign module
* localization module

Following figure further explains the structure, firstly input videos shall be divided into snippets, then features for each snippet shall be extracted by using GCNeXt blocks. Each GCNeXt contains two graph convolution streams, one stream is responsible for fixed temporal neighbors, the other combine context into features for snippet. At the end of GCNeXt blocks, sub-graphs are extracted. Then these sub-graphs are represent by SGAlign(sub-graph of interest alignment) module. Finally localization module shall sort the detection scores and gain corresponding predicted start and end time.
![](https://github.com/Atherzzz/gtad_reproduce_project/blob/main/g-tad_structure.PNG)

## Dataset
### THUMOS-14
The original dataset used by the paper is  [THUMOS 2014](http://crcv.ucf.edu/THUMOS14) used for action recognition. It contains 30 hours of 413 temporally annotated, untrimmed video from [20 actions categories classes](http://crcv.ucf.edu/THUMOS14/Class%20Index_Detection.txt). From which, 200 are used on the validation set for training and the 213 remaining videos are used for evaluation on the testing set. 
In fact, in the paper, features are pre-extracted and the dataset pretrained. Which makes it hard to apply this on a new dataset.
### MultiTHUMOS
The dataset [MultiTHUMOS](https://ai.stanford.edu/~syyeung/everymoment.html) extends the annotation of Thumos 14 with 65 classes instead of 20 for THUMOS14 for an average of 1.5 labels per frame instead of 0.3 for thumos14. The new dataset MultiTHUMOS introduces multilabeling, as in fact in most video, multi actions are realized simultaneously, for example we could be walking while drinking coffee. 
| Characteristic | Thumos 14 | MultiThumos |
| -------- | -------- | -------- |
|Number of classes|20|65|
| Density of annotation (on average) | 0.3 | 1.5 |
|  Maximum number of actions per frame|  2|9  |
| Maximum number of actions per video | 3 | 25 |
| Range of annotation per action classes| 3.7 minutes to 1 hour | 27 seconds to 5 hours |
| Average length of instance of action class | 4.8 seconds|  3.3 seconds|

Here are some examples of the characteristic that MultiThumos changes : more classes, more annotation but instances are shorter in average and actions classes are less uniformly annotated. 

## Code Implementation
Firstly, we tried to implement the original code locally, however as both of us lack powerful GPU, so we migrate our code on Google Colab, found out even we use a lower batch size on free default GPU provided by Colab, it is still time-consuming  and out of memory.
So both of us have to upgraded to Colab pro for a higher memory GPU, and it cost 3 hours to obtain the result(2 hours for training, and 15 minutes for interfacing, and 1 hour for postprocessing respectively)
You can check [our code](https://github.com/Atherzzz/gtad_reproduce_project) here 
You can check [here](https://github.com/frostinassiky/gtad) for original code.
There are two branches in our GitHub, [main](https://github.com/Atherzzz/gtad_reproduce_project/tree/main) branch contains new MultiTHUMOS dataset, also the blog and script, [Code](https://github.com/Atherzzz/gtad_reproduce_project/tree/Code) branch contains our code.  

For the reproduction task, we achieve the expected results stated in the official implementation GitHub for this project : G-TAD.
## Our contribution
### Dataset migration
We are going to use the [MultiTHUMOS dataset](https://ai.stanford.edu/~syyeung/everymoment.html) on the action detection task.  
Basically, what I did was from the multiTHUMOS dataset (class_list and annotation from each class), I split all annotation in two, one for test videos and the other one for train videos. 
Since the multiTHUMOS didn't contains all information needing in this project, such as frames information, I had to extrapolate them from the original paper on THUMOS14 annotations. In fact for validations videos, I got the frames rates information manually and included them manually. 
We also had to update the thumos_gt json file, from the multiTHUMOS dataset, which contains for each validation or test video their labels and their respective start time and end time. 
The code to performs this migration is available in our [git repository](https://github.com/Atherzzz/gtad_reproduce_project/tree/main/MultiTHUMOS) as well as the new annotations and evaluation file generated from the MultiTHUMOS dataset. 

### Hyperparameter Tuning
There are mainly eight hyperparameter shown on original code, which includes:
* Learning Rate
* Weight Decay
* Train Epochs
* Batch Size
* Step Size
* Step Gamma
* Number of GPU
* Number of CPU


Due to the limitation of Colab, we can not tune the last two hyperparameters, in the rest hyperparameters, we choose to tune learning rate, batch size and weight decay, which are more general during finetuning. Due to the long time consuming, we test the performance on different hyperparameters from 25% of default setting to no more than 200% of default setting. 

For batch size, as we do not hold enough memory for batch size higher than 8, we have to test its performance under 8. 



For Learning Rate, we test its performance from 0.00002 to 0.0001, these thresholds are found based on half & 2.5 times of default setting.


For Weight Decay, we found even we adjust it to four times of default setting, results on different MAP threshold hod different performance, so we generally keep this hyperparameter as default.
## Results
### Dataset migration
In fact, after making the dataset migration on MultiThumos, we obtain very low mAP% for the different tIoU. 
| Dataset | IoU@0.3 | IoU@0.4 | IoU@0.5 | IoU@0.6 | IoU@0.7 | 
| -------- | -------- | -------- | -------- | -------- | -------- |
|THUMOS'14 | 57.5 | 50.9 |42.6|32.3|22.9|
|MultiTHUMOS|0.29|0.25 | 0.19 |0.15| 0.10|

We did expected to achieve lower result on MultiTHUMOS but these indicate a failure in  action recognition with GTAD for MultiThumos, as least without further changes. 

After applying this dataset to lower thresholds, we achieve better results.
| Dataset | IoU@0.05 | IoU@0.1 | IoU@0.15 | IoU@0.2 | IoU@0.25 | 
| -------- | -------- | -------- | -------- | -------- | -------- |
|THUMOS'14 | 65.3| 64.8 |63.5|62.3|60.4|
|MultiTHUMOS|17.0| 15.3 |13.8 |12.8| 11.8|

In fact, this model has to classify more than 3 times more classes for multiTHUMOS compared to THUMOS14 which could explain lower results for multiTHUMOS. Moreover, some actions classes only have few annotations in duration and each instance is shorter in average for multiTHUMOS. 

### Hyperparameter Tuning
Following we show the result after we tunning three parameters(Batch Size, Learning Rate, and Weight Decay), the bold parameters hold the default setting in original paper. Through our work, we found out even original parameters hold best performance on three MAP scores, our fine-tuned hyperparameter improved the performance of model under MAP@0.6 and MAP@0.7, which are also bold.
| Batch Size | Learning Rate |  Weight Decay | MAP@0.3 | MAP@0.4 | MAP@0.5 |MAP@0.6 |MAP@0.7 |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 8     | 0.0001     | 1.00E-04   | 0.556     | 0.495     | 0.420    |0.322    |0.216    |
| 8     | 0.00006     | 1.00E-04   | 0.556     | 0.504     | 0.424     |**0.329**     |**0.230**     |
| 8     | 0.00006     | 2.00E-04     | 0.566     | 0.504     | 0.425    |0.325     |0.229     |
| 8     | 0.00006      | 4.00E-04     |0.565     | 0.502     | 0.424    |0.319     |0.226     |
| **8**    | **0.00004**    | **1.00E-04**    | **0.576**    | **0.509**    | **0.427**     |0.320     |0.228     |
| 4     | 0.00004    | 1.00E-04      | 0.566     | 0.500     | 0.425    |0.328     |0.225     |
| 2    | 0.00004    | 1.00E-04      | 0.548     | 0.490     | 0.412     |0.317     |0.219    |
| 8    | 0.00002    | 1.00E-04     | 0.555     | 0.486     | 0.402     |0.302     |0.206     |
| 4     | 0.00002     | 1.00E-04      | 0.545     | 0.487    | 0.396     |0.298    |0.196     |
| 8     | 0.00001    | 1.00E-04     | 0.519     | 0.451     | 0.370     |0.263    |0.169     |

## Reflection
### Meng Zheng
This project give me a chance to actually reproduce an original paper, from this project, I learned more about pytorch, also the details of G-TAD. Also through fine-tuning this model, I learned something more about how to tune these parameters.Through working with Annie, I think I also improved my communication skills. During the reproduction, it always takes me a lot of time to understand the source code, so in the future I will try to reproduce more papers to improve this.
### Annie Liu
This project was very interesting to reproduce even I couldn't understand every detail of the project and the code but only a globality.  
In fact, I had never dealt with code and data that are so time-consuming to compute. Eventually it only took us 3 hours for each run using colab Tesla P100 GPU, but it would have taken a day for each time otherwise. It made me aware that it was even more important to double check our changes before running it since it was consuming if we did not want to have rerun the whole process for small inattention mistakes.  
Also, originally, I had planned to realize data augmentation with making changes to original THUMOS14 videos but it turned to be more complicate to apply GTAD on a new dataset since it uses pretrained data. Thanks to the suggestion of our TA, Attila Lengyel, to try multiTHUMOS, I also discovered new ways to deal with data : with the labels in itself. It was really nice to work with Meng and it allowed me to improve my teamwork skills. 


## References
* [1] Mengmeng Xu, Chen Zhao, David S. Rojas, Ali Thabet, and  Bernard Ghanem. G-TAD: Sub-Graph Localization for Temporal Action Detection. 2020.
* [2] Serena Yeung, Olga Russakovsky, Ning Jin, Mykhaylo Andriluka, Greg Mori, and Li Fei-Fei. Every Moment Counts: Dense Detailed Labeling of Actions in Complex Videos. 2015.
