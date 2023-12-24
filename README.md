# ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-
The dataset consists of 102 flower categories


## Project Overview
This project is part of the Udacity PyTorch Scholarship Challenge and focuses on deep learning for multilabel image classification. The challenge details can be found on its [official website](https://sites.google.com/udacity.com/pytorch-scholarship-facebook/phase-1-archived/phase-1-home).

As the final project for the Challenge, the classification of 102 flower species from the [Visual Geometry Group University of Oxford dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102) was chosen.

The dataset comprises 102 flower categories, with each class containing between 40 and 258 images of approximately 500x600 pixels. The chosen flowers are commonly found in the United Kingdom. The images exhibit large-scale variations, pose changes, and varying lighting conditions. Additionally, there are categories with significant variations within the category and several closely related categories.

An unofficial project leaderboard is available on [Kaggle](https://www.kaggle.com/c/oxford-102-flower-pytorch/leaderboard).

## Methods Used
I employed ResNet architectures of various sizes for the classification task. To enhance training, different learning rate schedulers were explored, including [OneCycleLR](https://arxiv.org/abs/1803.09820), [CyclicLR](https://arxiv.org/pdf/1506.01186.pdf), and [WarmRestartsLR](https://arxiv.org/abs/1608.03983).

The optimization process utilized a modified version of the Adam optimizer, known as [AdamW](https://www.fast.ai/2018/07/02/adam-weight-decay/). The standard ResNet models from PyTorch were adjusted using a set of techniques inspired by the [fast.ai library](https://github.com/fastai/fastai).

To further improve performance, test time augmentation (TTA) and the combination of multiple models trained using cross-validation were employed. Surprisingly, achieving top-notch performance was possible even with a relatively small ResNet34 model, and the training time for such a model was less than 20 minutes on a Tesla K80 GPU.

## Results
The project achieved an impressive **99.5%** accuracy on the test dataset by using 5 ResNet101 models trained using cross-validation. However, noteworthy accuracy of **97%** was also attainable with smaller networks like ResNet34.

astai

Resnet32, 5+5 epochs, 64 bs, lr=0.02
epoch	train_loss	valid_loss	accuracy
5	    0.059245	0.104859	0.977995


Resnet152, 5+5 epochs, 32 bs, lr=0.01
epoch	train_loss	valid_loss	accuracy
5	    0.036988	0.105719	0.986553

Densenet121, 5+5 epochs, 32 bs, lr=0.02
epoch	train_loss	valid_loss	accuracy
5	    0.046098	0.097080	0.981663


pytorch

0.25 subset

resnet34, 15 epochs, lr=0.01/0.001 StepRL schedule 5, basic affine transforms, 45 deg rot
Best val Loss: 0.429578, Best val Acc: 0.891198
Best val Loss: 0.354743, Best val Acc: 0.922983

full dataset

resnet34, 15 epochs
Best val Acc: 0.863081
Best val Acc: 0.900978

resnet34, 15 epochs, lr=0.01/0.001 StepRL schedule 5, basic affine transforms, 45 deg rot
Best val Loss: 0.248217, Best val Acc: 0.941320
Best val Loss: 0.131558, Best val Acc: 0.974328

resnet34, 15 epochs, 1cycle + other tricks from fastai, basic affine transforms, 45 deg rot
Best val Loss: 0.246839, Best val Acc: 0.943765
Best val Loss: 0.093415, Best val Acc: 0.982885

resnet34, 15 epochs, 1cycle + other tricks from fastai, AdamW wd=0.01/0.1, basic affine transforms, 45 deg rot
Best val Loss: 0.230131, Best val Acc: 0.948655
Best val Loss: 0.078676, Best val Acc: 0.987775

resnet34, 15 epochs, 1cycle lr=1e-2/3e-4 + other tricks from fastai, AdamW wd=0.01/0.01, basic affine transforms, 45 deg rot, 10xTTA
Best val Loss: 0.138446, Best val Acc: 0.965770
Best val Loss: 0.068852, Best val Acc: 0.987775

resnet101, 15 epochs, bs 48/64, 1cycle lr=3e-3/7e-5 + other tricks from fastai, AdamW wd=5e-2/2e-4, basic affine transforms, 45 deg rot, 10xTTA
Best val Loss: 0.111695, Best val Acc: 0.968215
Best val Loss: 0.054569, Best val Acc: 0.992665

resnet152, 15 epochs, bs 48/64, 1cycle lr=3e-3/7e-5 + other tricks from fastai, AdamW wd=5e-2/2e-4, basic affine transforms, 45 deg rot, 10xTTA
Best val Loss: 0.114294, Best val Acc: 0.976773
Best val Loss: 0.069883, Best val Acc: 0.991443

resnet34, 15 epochs, pretrain on 128, 1cycle lr=1e-2/3e-4/1e-2/1e-4 + other tricks from fastai, AdamW wd=0.01/0.01/0.01/0.1, basic affine transforms, 45 deg rot, 10xTTA
Best val Loss: 0.193959, Best val Acc: 0.943765
Best val Loss: 0.084297, Best val Acc: 0.982885
Best val Loss: 0.093187, Best val Acc: 0.988998
Best val Loss: 0.098492, Best val Acc: 0.990220

5CV

resnet50, 15 epochs, 1cycle lr=1e-2/1e-4 + other tricks from fastai, AdamW wd=0.01/0.1, basic affine transforms, 45 deg rot, 10xTTA
mean_acc	    0.987042
mean_acc_head	0.964303
mean_loss	    0.082470
mean_loss_head	0.147141

resnet101, 15 epochs, bs 48/64, 1cycle + other tricks from fastai, lr=3e-3/7e-5, AdamW wd=5e-2/2e-4, basic affine transforms, 45 deg rot, 10xTTA
mean_acc	    0.988467
mean_acc_head	0.969335
mean_loss	    0.059509
mean_loss_head	0.127831

![2023-12-24_075228](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/b2a9a107-f4ba-49c8-b783-eeeec05845ca)


![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/6450af13-1fd6-4edd-ba44-1d94a23db975)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/7bd89f3b-cda1-469c-9928-e96fd3a55a04)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/45f540d1-a9fe-4eac-97c5-45d3dc0ef609)
 
Epoch 1/3
----------
.........................................................................................................................................
train Loss: 3.1069 Acc: 0.3385
..................
valid Loss: 0.6397 Acc: 0.8778

Epoch 2/3
----------
.........................................................................................................................................
train Loss: 0.8014 Acc: 0.7908
..................
valid Loss: 0.3073 Acc: 0.9364

Epoch 3/3
----------
.........................................................................................................................................
train Loss: 0.5096 Acc: 0.8675
..................
valid Loss: 0.2626 Acc: 0.9499

Training complete in 1m 33s
Best val Loss: 0.262621, Best val Acc: 0.949878

Epoch 1/5
----------
.......................................................................................................
train Loss: 0.3996 Acc: 0.8985
.............
valid Loss: 0.1891 Acc: 0.9621

Epoch 2/5
----------
.......................................................................................................
train Loss: 0.2487 Acc: 0.9380
.............
valid Loss: 0.1565 Acc: 0.9633

Epoch 3/5
----------
.......................................................................................................
train Loss: 0.1679 Acc: 0.9582
.............
valid Loss: 0.1141 Acc: 0.9817

Epoch 4/5
----------
.......................................................................................................
train Loss: 0.1239 Acc: 0.9722
.............
valid Loss: 0.0942 Acc: 0.9853

Epoch 5/5
----------
.......................................................................................................
train Loss: 0.0929 Acc: 0.9818
.............
valid Loss: 0.0908 Acc: 0.9890

Training complete in 2m 21s
Best val Loss: 0.090757, Best val Acc: 0.988998
Epoch 1/15
----------
.......................................................................................................
train Loss: 0.0071 Acc: 0.9988
.............
valid Loss: 0.0534 Acc: 0.9914

Epoch 2/15
----------
.......................................................................................................
train Loss: 0.0075 Acc: 0.9980
.............
valid Loss: 0.0566 Acc: 0.9890

Epoch 3/15
----------
.......................................................................................................
train Loss: 0.0084 Acc: 0.9979
.............
valid Loss: 0.0572 Acc: 0.9890

Epoch 4/15
----------
.......................................................................................................
train Loss: 0.0062 Acc: 0.9989
.............
valid Loss: 0.0549 Acc: 0.9914

Epoch 5/15
----------
.......................................................................................................
train Loss: 0.0070 Acc: 0.9986
.............
valid Loss: 0.0555 Acc: 0.9902

Epoch 6/15
----------
.......................................................................................................
train Loss: 0.0058 Acc: 0.9992
.............
valid Loss: 0.0541 Acc: 0.9914

Epoch 7/15
----------
.......................................................................................................
train Loss: 0.0090 Acc: 0.9983
.............
valid Loss: 0.0535 Acc: 0.9914

Epoch 8/15
----------
.......................................................................................................
train Loss: 0.0064 Acc: 0.9989
.............
valid Loss: 0.0561 Acc: 0.9902

Epoch 9/15
----------
.......................................................................................................
train Loss: 0.0065 Acc: 0.9986
.............
valid Loss: 0.0551 Acc: 0.9902

Epoch 10/15
----------
.......................................................................................................
train Loss: 0.0083 Acc: 0.9985
.............
valid Loss: 0.0567 Acc: 0.9914

Epoch 11/15
----------
.......................................................................................................
train Loss: 0.0063 Acc: 0.9991
.............
valid Loss: 0.0553 Acc: 0.9890

Epoch 12/15
----------
.......................................................................................................
train Loss: 0.0074 Acc: 0.9983
.............
valid Loss: 0.0550 Acc: 0.9914

Epoch 13/15
----------
.......................................................................................................
train Loss: 0.0074 Acc: 0.9991
.............
valid Loss: 0.0561 Acc: 0.9902

Epoch 14/15
----------
.......................................................................................................
train Loss: 0.0064 Acc: 0.9986
.............
valid Loss: 0.0554 Acc: 0.9902

Epoch 15/15
----------
.......................................................................................................
train Loss: 0.0059 Acc: 0.9989
.............
valid Loss: 0.0546 Acc: 0.9914

Training complete in 12m 27s
Best val Loss: 0.053407, Best val Acc: 0.991443

![2](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/7df406fd-fb1f-4b6f-9695-e2139877f8cb)

 ![3](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/b19674a2-846d-48b6-a651-17b13620df07)
   

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/35f600cc-35e7-47cc-a99d-567e7be8ed76)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/43c80b2d-47f6-4de4-a94b-b87ac3c7e221)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/2c087e2b-6100-492a-903e-22026bc13c2d)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/cccdd369-18ac-464a-becb-173b61f4c593)
![4](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/b0549b16-1b87-422c-85b5-ed68fe96c7d4)

![image](https://github.com/KAMAlhameedawi/ResNet-Model-Based-Deep-Learning-Classification-of-102-Flower-Species-/assets/149914341/df1fd90a-6771-45b0-8864-7f3feb509b79)




