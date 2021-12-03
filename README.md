# Resilient UAV Swarm Communications with Graph Convolutional Neural Network 
This repository contains the source codes of 

[Resilient UAV Swarm Communications with Graph Convolutional Neural Network](https://arxiv.org/abs/2106.16048)

Zhiyu Mou, Feifei Gao, Jun Liu, and Qihui Wu

Fei-Lab

## Problem Descriptions
This work propose a meta GCN algorithm to realize the fast connectivity restore of initially connected robotic networks (RNs) with random nodes destroyed afterwards. Specifically, we first propose a GCO to heal the connectivity of the RNs and prove its convergence with contracting mapping. We then extend the GCO to a GCN with a carefully designed Lagrange-form loss function. The GCN is trained to find the healing topology of the RNs. Thirdly, we utilize the meta learninig scheme to find potential parameters for GCN, which can speed up its on-line training. In addition, we also consider the general destructions to the RNs and propose an efficient algorithm to restore the connectivity of RNs based on the proposed meta GCN.

## Display of Main Results Demo
### One-off UEDs
randomly destruct 150 robots (e.g. UAVs) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;     randomly destruct 100 robots (e.g.UAVs)

<img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/one_off_destruct_150.gif" width="320" alt="150">  <img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/one_off_destruct_100.gif" width="320" alt="100">

### General UEDs
general UEDs with global information &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;    general UEDs with monitoring mechanism

<img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/general_UEDs_global_info_low_quality.gif" width="320" alt="general_global_info"> <img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/general_UEDs_low_quality.gif" width="320" alt="general">

> Note: these are gifs. It may take a few seconds to display. You can refresh the page if they cannot display normally. Or you can view them in [./video](./video).
## Environment Requirements
> pytorch==1.6.0  
> torchvision==0.7.0   
> numpy==1.18.5  
> matplotlib==3.2.2  
> pandas==1.0.5  
> seaborn==0.10.1   
> cuda supports and GPU acceleration

Note: other versions of the required packages may also work.

The machine we use
> CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz  
> GPU: NVIDIA GeForce RTX 3090

## Necessary Supplementary Downloads 
As some of the necessary configuration files, including .xlsx and .npy files can not be uploaded to the github, we upload these files to the clouds. Anyone trying to run these codes need to download the necessary files.  
### Download initial UAV positions (necessary)
> To make the codes reproducible, you need to download the initial positions of UAVs we used in the experiment from https://cloud.tsinghua.edu.cn/f/c18807be55634378b30f/ or https://drive.google.com/file/d/1q1J-F2OAY_VDaNd1DWCfy_N2loN7o1XV/view?usp=sharing. Upzip the download files to [./Configurations/](./Configurations).
### Download Trained Meta Parameters (alternative, but if using meta learning without training again, then necessary)
> Since the total size of meta parameters is about 1.2GB, we have uploaded the meta parameters to https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ and https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing. You need to download the file from either two links and unzip them to [./Meta_Learning_Results/meta_parameters/](./Meta_Learning_Results/meta_parameters/)if you want to use the trained meta parameters. Otherwise, you need to train the meta parameters again (directly run [Meta-learning_all.py](./Meta-learning_all.py))
### Download Meta Learning Loss Functions Pictures (alternative)
> The loss function pictures of meta learning are available on https://cloud.tsinghua.edu.cn/f/fc0d84f2c6374e29bcbe/ and https://drive.google.com/file/d/1cdceleZWyXcD1GxOPCYlLsRVTwNRWPBy/view?usp=sharing. You can store them in [./Meta_Learning_Results/meta_loss_pic/](./Meta_Learning_Results/meta_loss_pic/)

## Quick Start
### Simulate SCC under one-off UEDs
directly run ./[Experiment_One_off_UED.py](./Experiment_One_off_UED.py)
```python
python Experiment_One_off_UED.py
```
### Simulate meta learning process
directly run ./[Meta-learning_all.py](./Meta-learning_all.py)
```python
python Meta-learning_all.py
```
### Simulate SCC under general UEDs
directly run ./[Experiment_General_UED.py](./Experiment_General_UED.py)
```python
python Experiment_General_UED.py
```
## File and Directory Explanations
* ./Configurations/  
> the initial positions of 200 UAVs
* ./Drawing/  
> the drawing functions
* ./Experiment_Fig/  
> the experiment figures and the drawing source codes
* ./Main_algorithm_GCN/  
> the proposed algorithms in the paper
> * ./Main_algorithm_GCN/CR_MGC.py
> > the CR-MGC algorithm (Algorithm 2 in the paper)
> * ./Main_algorithm_GCN/GCO.py
> > the GCO algorithm
> * ./Main_algorithm_GCN/Smallest_d_algorithm.py
> > algorithm of finding the smallest distance to make the RUAV graph a CCN (Algorithm 1 in the paper)
* ./Meta_Learning_Results/  
> the results of meta learning
> * ./Meta_Learning_Results/meta_loss_pic
> > the loss function pictures of 199 mGCNs
> * ./Meta_Learning_Results/meta_parameters
> > the meta parameters (Since the total size of meta parameters is about 1.2GB, we have uploaded the meta parameters to https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ or https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing)
* ./Traditional_Algorithm/  
> the implementations of traditional algorithms
* ./video/
> the gif files of one-off UEDs
* ./Configurations.py
> the simulation parameters
* ./Environment.py
> the Environment generating UEDs
* ./Experiment_General_UED.py/
> the simulation under general UEDs
* ./Experiment_One_off_UED.py/
> the simulation under one-off UEDs
* ./Experiment_One_off_UED_draw_Fig_12_d.py/
> draw the Fig. 12(d) in the simulation under one-off UEDs
* ./Meta-learning_all.py/
> the meta learning
* ./Swarm.py/
> the integration of algorithms under one-off UEDs
* ./Swarm_general.py/
> the integration of algorithms under general UEDs
* ./Utils.py/
> the utility functions
> 
*Note that some unnecessary drawing codes used in the paper are not uploaded to this responsitory.*
