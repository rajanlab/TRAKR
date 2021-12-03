## TRAKR

This repo contains basic code to train, test and classify time series patterns using TRAKR.

For details, see: TRAKR - A reservoir-based tool for fast and accurate classification of neural time-series patterns

https://www.biorxiv.org/content/10.1101/2021.10.13.464288v1

##

I also reuse/ reimplement some code from Fawaz et al 2019 related to their paper and repo as below:

 https://link.springer.com/article/10.1007/s10618-019-00619-1 
 
 https://github.com/hfawaz/dl-4-tsc

##

We provide Python implementations here. 

All software distributed under GNU GPL (see LICENSE.md).

A description of different files within this repository is provided below:

##

- <constants.py> - reused from Fawaz et al 2019

- <dtw.py> - contains an implementation of DTW from the following paper:

Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, Brandon Westover, Qiang Zhu, Jesin Zakaria, and Eamonn Keogh. 2012. Searching and mining trillions of time series subsequences under dynamic time warping. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '12). Association for Computing Machinery, New York, NY, USA, 262–270. DOI:https://doi.org/10.1145/2339530.2339576

- <esn.py> - implementation of twiESN from Fawaz et al 2019 

- <euclidean.py> - Euclidean distance for the classification of time series patterns

- <main.py> - main TRAKR file for the classification of time series patterns

- <mlp.py> - MLP for the classification of patterns - based on Fawaz et al 2019

- <modules.py> - custom functions used for different types of training, metric calculations etc

- <mutualinfo.py> - implements mutual info for the classification of patterns

- 

