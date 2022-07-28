## Introduction

This code was written by Nathaniel Bowers as part of the 2022 Stony Brook University REU. The project's goal was to update and improve the codebase found in Nanophotonic Particle Simulation and Inverse Design Using Artificial Neural Networks (doi/10.1126/sciadv.aar4206). This code was written to run on the the simulation data found in that paper; however, in theory, the network can work for any regression problem. The data file in the github was generated with the MatLab code from their paper.

Currently methods for making and optimizing the neural network have been written. Further work on this project would involve using the NN to solve inverse design problems related to metamaterials.

This code has been written and tested in python 3.10.4. See requirements.txt for the required python packages and versions.

## Data Preparation

It is assumed that the data fed into the neural network will consist of ordered $n$ pairs of input vectors $\vec{x}$ and output vectors $\vec{y}$ of arbitrary dimension. Similarly, it is assumed that the x- and y-data are stored in two separate csv files. A sample file may look like $[\vec{x}_1, \vec{x}_2,...,\vec{x}_n]$.

Data is loaded with the TrainingData class so the original csv files will only need to be read once. However, this does require enough memory to store the entirety of the x- and y- data. Once loaded, the x- data is normalized. Any data fed into the final model MUST be normalized with the same mean and std as the initial data. 

## Code Structure

After loading the data, it is packed with the hyperparameters and fed into the tuning function. That function then calls the network training function. After the network has been tuned, the ideal hyperparameters found are passed into the network training function and the final model is saved. Data on the performance of the tuning and final model are saved in the path given by 'output_path'. 

By commenting out the lines in main referring to tuning and passing a user set config dict, the tuning step can be skipped. 

## Using HPC

It is highly recommended to use some form of high performance computing when tuning the network. Tuning is both memory intensive and can benefit significantly from multiprocessing. See 'demo_run.sh' for an example of tuning the network on a computing cluster using slurm. 