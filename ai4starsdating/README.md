# AI for dating stars: a benchmarking study for gyrochronology

This repo contains data and code accompaning the paper, INCLUIR PAPER. It includes code for running all regression models developed for each Benchmark.

## Data

The folder `datasets` contains the two main datasets of the project:

- gyro_tot_v20180801.txt:  
  
  Data sample of 1464 stars with accurate ages coming from asteroseismology or cluster belonging. Used to perform the training of the models of all Benchmarks and testing these  models in Benchmarks A and B.  

- est_gyro.txt:  
  
  Control data sample of novel non-clustered 32 stars, including the Sun, to examine the age estimation performance of all the models in the Benchmark C.


## Dependencies
This code requires the following: 
- python 3.*
- scikit-learn 0.24.*

## Usage instructions
* Benchmark A:  
python ai4stellarage_Benchmark_A.py

* Benchmark B:  
python ai4stellarage_Benchmark_B1.py  
python ai4stellarage_Benchmark_B2.py

* Benchmark C:  
python ai4stellarage_Benchmark_C.py


