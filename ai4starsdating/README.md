# AI for dating stars: a benchmarking study for gyrochronology

This repo contains data and code accompaning the paper, INCLUIR PAPER. It includes code for running all regression models developed for each Benchmark.

### Content

  * [Citation](#citation)  
  * [Dependencies](#dependencies)
  * [Data](#data)
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
This code requires the following: 
- python 3.*
- scikit-learn 0.24.*

### Data

The folder `datasets` contains the two main datasets of the project:

- gyro_tot_v20180801.txt:  
  
  Data sample of 1464 stars with accurate ages coming from asteroseismology or cluster belonging. Used to perform the training of the models of all Benchmarks and testing these  models in Benchmarks A and B.  

- test_gyro.txt:  
  
  Control data sample of novel non-clustered 32 stars, including the Sun, to examine the age estimation performance of all the models in the Benchmark C.


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

