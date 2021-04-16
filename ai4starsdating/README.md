# AI for dating stars: a benchmarking study for gyrochronology

This repo contains data and code accompaning the paper, INCLUIR PAPER. It includes code for running all regression models developed for each Benchmark.

### Content

  * [Citation](#citation)  
  * [Dependencies](#dependencies)
  * [Data](#data)
  * [Code](#code)
  * [Usage](#usage)
    * [Training and testing](#training-and-testing)
    * [Testing with pretrained models](#testing-with-pretrained-models)
  * [Results](#results)
  * [Contact](#contact)


### Citation

If you find anything of this repository useful for your projects, please consider citing our work:

```bibtex
@inprocceedings{ai4dscvpr2021w,
	author  = {A. Moya and J. {Recio-Marti\'inez} and R.~J. {L\'opez-Sastre}},
	title   = {AI for dating stars: a benchmarking study for gyrochronology},
  	booktitle = {1st Workshop on AI for Space, CVPR},
	year	= {2021},	
}
```

### Dependencies
The project has been developed with the following packages: 
- python 3.8
- numpy 1.19.5
- pandas 1.2.1
- matplotlib 3.3.3
- scikit-learn 0.24.1


### Data

The folder `datasets` contains the two main datasets of the project:

- `gyro_tot_v20180801.txt`:  
  
  Data sample of 1464 stars with accurate ages coming from asteroseismology or cluster belonging. Used to perform the training of the models of all Benchmarks and testing these  models in Benchmarks A and B.  

- `test_gyro.txt`:  
  
  Control data sample of novel non-clustered 32 stars, including the Sun, to examine the age estimation performance of all the models in the Benchmark C.


### Code

The folder `code` contains the following scripts:

- `ai4stellarage_Benchmark_A.py`:  
  Script to evaluate the different regression models following a classical training/test data splitting scheme. From the data sample distribution `datasets/gyro_tot_v20180801.txt`, we release a training set and a testing set, where 80 % and 20 % of the stars have been randomly included, respectively.

- `ai4stellarage_Benchmark_B1.py`:  
  Script to evaluate the generalization capability of models, where we train the approaches on young stars, and evaluate their performance on old stars.  
  
- `ai4stellarage_Benchmark_B2.py`:  
  Script of a second scenario to evaluate the generalization capability of the models when they are trained only with stars belonging to clusters or only with field stars.

- `ai4stellarage_Benchmark_C.py`:  
  Script to examine the age estimation performance of all the models on a control data sample `datasets/test_gyro.txt`, composed only of stars not belonging to any cluster, and with a more realistic age distribution.


### Usage  

#### Training and testing  
  
To reproduce our results, training and testing our models, you just have to run the following command, taking into account the desire Benchmark:

```bash
python ai4stellarage_Benchmark_A.py
python ai4stellarage_Benchmark_B1.py  
python ai4stellarage_Benchmark_B2.py
python ai4stellarage_Benchmark_C.py
```

#### Testing with pretrained models



### Results


### Contact

For any question, you can open an issue or contact:

- Jarmi Recio Martínez: jarmi.recio@edu.uah.es

