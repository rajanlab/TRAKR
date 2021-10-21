# TRAKR

This repo contains basic code to train and test time series patterns using TRAKR.

For details, see: TRAKR - A reservoir-based tool for fast and accurate classification of neural time-series patterns https://www.biorxiv.org/content/10.1101/2021.10.13.464288v1

We provide Python implementations here. 

All software distributed under GNU GPL (see LICENSE.md).

A description of different files within this repository is provided below:

---------------------------------------------------------------------------


#########################

trakr_modules.py

- main trakr function, which performs one shot training and testing

#########################

trakr_simple.py

- the simplest version of trakr which can help you detect changes in a single pattern
- should be able to just run

#########################

trakr_patternchange_syntheticsignals.py

- A remake of figure 2 A and B (separately) from the paper essentially
- Should be just able to run

#########################


trakr_seqmnist.py

- does trakr classification on sequential MNIST data
- should be able to just run

#########################


trakr_macaqueofc.py

- does trakr classification on macaque OFC data
- The OFC data is available upon request

#########################

comparisonmethods.py

- different comparison methods to classify MNIST digits, as in the paper
- uncomment each method to run it

#########################






