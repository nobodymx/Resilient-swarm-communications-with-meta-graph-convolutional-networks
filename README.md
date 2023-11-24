# Resilient UAV Swarm Communications with Graph Convolutional Neural Network 
This repository contains the source codes of 

[Resilient UAV Swarm Communications with Graph Convolutional Neural Network](https://arxiv.org/abs/2106.16048)

Zhiyu Mou, Feifei Gao, Jun Liu, and Qihui Wu

Fei-Lab

## Problem Descriptions
This work proposes a meta GCN algorithm to realize the fast connectivity restore of initially connected robotic networks (RNs) with random nodes destroyed afterward. Specifically, we first propose a GCO to heal the connectivity of the RNs and prove its convergence with contracting mapping. We then extend the GCO to a GCN with a carefully designed Lagrange-form loss function. The GCN is trained to find the healing topology of the RNs. Thirdly, we utilize the meta-learning scheme to find potential parameters for GCN, which can speed up its online training. In addition, we also consider the general destructions to the RNs and propose an efficient algorithm to restore the connectivity of RNs based on the proposed meta GCN.

## Main Results Demo
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
> Cuda supports and GPU acceleration

Note: other versions of the required packages may also work.

The machine we use
> CPU: Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz  
> GPU: NVIDIA GeForce RTX 3090

## Necessary Supplementary Downloads 
As some of the necessary configuration files, including .xlsx and .npy files can not be uploaded to GitHub, we upload these files to the cloud. Anyone trying to run these codes needs to download the necessary files.  
### Download initial UAV positions (necessary)
> To make the codes reproducible, you need to download the initial positions of UAVs we used in the experiment from https://cloud.tsinghua.edu.cn/f/c18807be55634378b30f/ or https://drive.google.com/file/d/1q1J-F2OAY_VDaNd1DWCfy_N2loN7o1XV/view?usp=sharing. Unzip the download files to [./Configurations/](./Configurations).
### Download Trained Meta Parameters (alternative, but if using meta-learning without training again, then necessary)
> Since the total size of meta parameters is about 1.2GB, we have uploaded the meta parameters to https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ and https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing. You need to download the file from either two links and unzip them to [./Meta_Learning_Results/meta_parameters/](./Meta_Learning_Results/meta_parameters/)if you want to use the trained meta parameters. Otherwise, you need to train the meta parameters again (directly run [Meta-learning_all.py](./Meta-learning_all.py))
### Download Meta Learning Loss Functions Pictures (alternative)
> The loss function pictures of meta-learning are available on https://cloud.tsinghua.edu.cn/f/fc0d84f2c6374e29bcbe/ and https://drive.google.com/file/d/1cdceleZWyXcD1GxOPCYlLsRVTwNRWPBy/view?usp=sharing. You can store them in [./Meta_Learning_Results/meta_loss_pic/](./Meta_Learning_Results/meta_loss_pic/)

## Quick Start
### Simulate SCC under one-off UEDs
directly run ./[Experiment_One_off_UED.py](./Experiment_One_off_UED.py)
```python
python Experiment_One_off_UED.py
```
### Simulate the meta-learning process
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
> The initial positions of 200 UAVs
* ./Drawing/  
> The drawing functions
* ./Experiment_Fig/  
> The experiment figures and the drawing source codes
* ./Main_algorithm_GCN/  
> The proposed algorithms in the paper
> * ./Main_algorithm_GCN/CR_MGC.py
> > The CR-MGC algorithm (Algorithm 2 in the paper)
> * ./Main_algorithm_GCN/GCO.py
> > The GCO algorithm
> * ./Main_algorithm_GCN/Smallest_d_algorithm.py
> > algorithm of finding the smallest distance to make the RUAV graph a CCN (Algorithm 1 in the paper)
* ./Meta_Learning_Results/  
> The results of meta-learning
> * ./Meta_Learning_Results/meta_loss_pic
> > The loss function pictures of 199 mGCNs
> * ./Meta_Learning_Results/meta_parameters
> > The meta parameters (Since the total size of meta parameters is about 1.2GB, we have uploaded the meta parameters to https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ or https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing)
* ./Traditional_Algorithm/  
> The implementations of traditional algorithms
* ./video/
> The gif files of one-off UEDs
* ./Configurations.py
> The simulation parameters
* ./Environment.py
> The environment generating UEDs
* ./Experiment_General_UED.py/
> The simulation under general UEDs
* ./Experiment_One_off_UED.py/
> The simulation under one-off UEDs
* ./Experiment_One_off_UED_draw_Fig_12_d.py/
> draw the Fig. 12(d) in the simulation under one-off UEDs
* ./Meta-learning_all.py/
> the meta-learning
* ./Swarm.py/
> The integration of algorithms under one-off UEDs
* ./Swarm_general.py/
> The integration of algorithms under general UEDs
* ./Utils.py/
> The utility functions
> 
*Note that some unnecessary drawing codes used in the paper are not uploaded to this repository.*
