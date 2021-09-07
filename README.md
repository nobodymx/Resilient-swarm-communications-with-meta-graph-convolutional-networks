# Resilient UAV Swarm Communications with Graph Convolutional Neural Network 
This repository contains the source codes of 

[Resilient UAV Swarm Communications with Graph Convolutional Neural Network](https://arxiv.org/abs/2106.16048)

Zhiyu Mou, Feifei Gao, Jun Liu, Ting Zhou, and Qihui Wu

Fei-Lab

## Problem Descriptions
In this paper, we study the self-healing of communication connectivity (SCC) problem of unmanned aerial vehicle (UAV) swarm network (USNET) that is required to quickly rebuild the communication connectivity under unpredictable external destructions (UEDs). Firstly, to cope with the one-off UEDs, we propose a graph convolutional neural network (GCN) and find the recovery topology of the USNET in an on-line manner. Secondly, to cope with general UEDs, we develop a GCN based trajectory planning algorithm that can make UAVs rebuild the communication connectivity during the self-healing process. We also design a meta learning scheme to facilitate the on-line executions of the GCN. Numerical results show that the proposed algorithms can rebuild the communication connectivity of the USNET more quickly than the existing algorithms under both one-off UEDs and general UEDs. The simulation results also show that the meta learning scheme can not only enhance the performance of the GCN but also reduce the time complexity of the on-line executions.

## Display of Main Results Demo
randomly destruct 150 UAVs 

<img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/one_off_destruct_150.gif" width="320" alt="150" "150">

randomly destruct 100 UAVs

<img src="https://github.com/nobodymx/resilient_swarm_communications_with_meta_graph_convolutional_networks/blob/main/video/one_off_destruct_100.gif" width="320" alt="100">

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

## Necessary Supplementary Download 
As some of the necessary configuration files, including .xlsx and .npy files can not be uploaded to the github, we upload these files to the clouds. Anyone trying to run these codes need to download the necessary files.  
### Download initial UAV positions (necessary)
> To make the codes reproducible, you need to download the initial positions of UAVs we used in the experiment to the ./Configurations/ from https://cloud.tsinghua.edu.cn/f/c18807be55634378b30f/ or https://drive.google.com/file/d/1q1J-F2OAY_VDaNd1DWCfy_N2loN7o1XV/view?usp=sharing
### Download Trained Meta Parameters (alternative, but if using meta learning without training again, then necessary)
> Since the total size of meta parameters is about 1.2GB, we have uploaded the meta parameters to https://cloud.tsinghua.edu.cn/f/2cb28934bd9f4bf1bdd7/ and https://drive.google.com/file/d/1QPipenDZi_JctNH3oyHwUXsO7QwNnLOz/view?usp=sharing. You need to download the file from either two links if you want to use the trained meta parameters. Otherwise, you need to train the meta parameters again (directly run [Meta-learning_all.py](./Meta-learning_all.py))
### Download Meta Learning Loss Functions Pictures (alternative)
> The loss function pictures of meta learning are available on https://cloud.tsinghua.edu.cn/f/fc0d84f2c6374e29bcbe/ and https://drive.google.com/file/d/1cdceleZWyXcD1GxOPCYlLsRVTwNRWPBy/view?usp=sharing

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
## File and Directory Explainations
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
