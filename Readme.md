# Introduction

This code is inspired by the github code of Olivier Sobrie (oso-pymcda => https://github.com/oso/pymcda). 
The latter proposes several methods to solve MCDA models inference problems (inferring MCDA models parameters from assignment examples).
Particularly, it proposes some metaheuristics and MIP algorithms to learn MR-Sort parameters considering monotone criteria (gain criteria).
This present code extends oso-pymcda in two ways.
It describes algorithms (metaheuristics and MIP formulation) to infer MR-Sort parameters first with single-peaked/single-valley criteria (a case of non monotone criteria), and secondly with potentially unknown criteria preference direction.
This code is made of 3 algorithms : 
- META : a metaheuristic to learn MR-Sort parameters from monotone criteria but unknown criteria preference direction (for more details, see https://hal.archives-ouvertes.fr/hal-03102714).
- MIP-SP : a Mixed Integer Programming formulation to learn MR-Sort parameters with single-peaked criteria (for more details, see https://arxiv.org/abs/2107.09668)
- META-SP : a metaheuristic to learn MR-Sort parameters with single-peaked criteria (adapted with large data sets sizes)



# Installation

It is recommanded to use a Linux environment.
1. This code was implemented in Python 3.7. Please check if you have the right version with this command on a terminal : python --version . If not, you can execute the following commands :
    - `sudo apt update`
    - `sudo apt upgrade`
    - `sudo add-apt-repository ppa:deadsnakes/ppa`
    - `sudo apt-get update`
    - `sudo apt-get install python3.7`
    - Open the  `~/.bashrc` file and write the following lines at the end of the file: `alias python=python3.7` and `alias python3=python3.7`. Then execute `source ~/.bashrc`. It must force python3.7 to be called in the place of python.
2. Install pip. It can be necessary to installl python3.7-distutils before. Execute the following commands :
    - `sudo apt install python3-pip`
    - `sudo apt-get install python3-distutils`
    - `sudo apt-get install python3.7-distutils`
    - `sudo apt-get update`
4. Download CPLEX Optimization Studio. Go to https://www.ibm.com/products/ilog-cplex-optimization-studio (choose the student/teacher free edition) and follow the steps until the download of the "ILOG CPLEX Optimization Studio" following your operating system. The CPLEX version used in this notebook is 12.9. You may have to create a IBMid account. The executable for Linux has the form of `cplex_studioXXX.bin`. Create a repertory called IBM in your home repertory. Copy `cplex_studioXXX.bin` into it. You may have to be granted permissions to execute `cplex_studioXXX.bin`, so execute `sudo chmod 777 cplex_studioXXX.bin`, then `./cplex_studioXXX.bin`. During the CPLEX installation, indicate the repository `/IBM` has the location of the source code of CPLEX.
5. Now, execute the following commands inside `/IBM/cplex/python/..`, where is located the file `setup.py` : 
    - `pip install docplex`
    - `sudo chmod 777 /usr/local/lib/python3.7/dist-packages`
    - `python3.7 setup.py install`
7. Set the environment variable PYTHONPATH on the terminal so that it may contains the absolute path to "IBM/cplex/python". Here is an example : `export PYTHONPATH=$PYTHONPATH:/home/IBM/cplex/python`
8. Inside the project package (`cd pm-pymcda`), execute the following commands : 
    - Add to the PATH variable, the absolute path to "IBM/cplex" (for example,`PATH=$PATH:/home/IBM/cplex`)
    - `source ~/.profile`
9. Install `numpy` library with this command : `python -m pip install numpy` .
10. Execute the baseline example of the use of learning algorithms : for instance for META, execute `python apps/meta.py` . Look at the results in `pm-pymcda/results_meta`.


# Package description

The package project is composed of 5 directories and 3 files at the root : 
* `readme.md`, the read-me file,
* `pymcda` : which contains contains back seat methods/algorithm and the definitions of elementary objects, particularly MR-Sort objects (defined in the `types.py`) 
    * `__init__.py` : which defines the module pymcda.
    * `electre_tri.py` : it mainly contains the definition of an MR-Sort, MCDA classes and their appropriate methods that are called during the creation of new models.
    * `generate.py` : it some contains functions useful for the generation of random instances and values (models, weights, profiles, etc ...).
    * `pt_sorted.py` : it is contains an adhoc class for the treatment of performance tables while maintaining an order in the evaluations.
    * `types.py` : it is contains the definitions of MCDA concepts as objects (criteria, alternatives, categories, etc ...).
    * `utils.py` : it is contains short methods essentially useful for the generation of noise in the learning set.
    * `learning` : a directory that contain core elements of algorithms pertaining to the learning process
        * `meta_mrsort.py` (resp. `meta_mrsort_sp.py` the equivalent which deals with single-peaked criteria) : encompasses the global structure of the metaheuristic (handling the population of model via a multithreading process, optimization of weights and profiles, iterative process and stopping conditions/fitness)
        * `lp_mrsort_weights_meta.py` (resp. `lp_mrsort_weights_meta_sp.py` the equivalent which deals with single-peaked criteria) : this is dedicated to the construction and resolution of the Linear Program in order to optimize weights and thresholds during the learning process.
        * `heur_mrsort_profiles_meta.py` (resp. `heur_mrsort_profiles_meta_sp.py` the equivalent which deals with single-peaked criteria) : this is dedicated to the initialization and update of profiles (through the choice of random and promising moves of the profiles) during the learning process.
        * `mip_mrsort_sp.py` : it is contains the complete formulation of the MIP and its resolution through CPLEX solver for the learning of MR-Sort parameters with potentially single-peaked criteria.
* `apps` : which contains essentially the implementation of the three learning algorithms.
* `meta.py` : the file that describes the META algorithm. It contains a class `RandMRSortMetaLearning` whose object is a test instance for the problem of learning MR-Sort models using an metaheuristic, from random assignment examples without knowing in advance the preference directions of criteria.
* `meta_sp.py` : the file that describes the META-SP algorithm. It contains a class `RandMRSortMetaSPLearning` whose object is a test instance for the problem of learning MR-Sort models with potentially single-peaked criteria using an metaheuristic, from random assignment examples.
* `mip_sp.py` : the file that describes the MIP-SP algorithm. It contains a class `RandMRSortMIPSPLearning` whose object is a test instance for the problem of learning MR-Sort models with potentially single-peaked criteria using a MIP formulation, from random assignment examples.



# Examples of execution

In each file describing each algorithm (meta.py, meta_sp.py, mip_sp.py) there is a main section (end of the file) that present a example of execution of the given algorithm.

The main part contains a set of parameters (problem, algorithm and tests parameters), that are already fixed.
These settings can be seen as a baseline which have in common : 5 criteria, 50 alternatives in the learning set and 2 categories.
These parameters can be tuned.
In this part, the problem/test parameters, as well as the parameters of the algorithm (META, META-SP, MIP-SP) are defined.
A variable "inst" is defined as a instance of test.
Finally such instance is executed by learning new models through the algorithm (META, META-SP, MIP-SP), and performing a generalization phase.
The results of these procedures are summarized into three kind of files :
- `instance_YYY-XXX.csv` : details of the performance table for one instance (only if the number of trials 'nb_models' is 1)
- `random_tests_details_YYY-XXX.csv` : details of the execution of each trial  (original and learned models, performances/metrics)
- `summary_results_YYY-XXX.csv` : recap of the results (execution time, classification accuracy (learning and test), preference direction restoration)
YYY denotes the type of algorithm (META, META-SP, MIP-SP), and XXX an identification number based on timestamp and/or details of the parameters of the test/problem.


## META : meta.py

The file to execute the metaheuristic that consists to learn the parameters of an MR-Sort model from assignment examples.
The algorithm assumes that the preference directions of criteria are unknown. 
Thus, it also infers the preference directions of unknown ones (`l_dupl_criteria`).

## MIP-SP : mip_sp.py

The file to execute the mixed integer programming algorithm (using CPLEX) to infer the parameters of an MR-Sort model with single-peaked/single-valley criteria from assignment examples
The algorithm deals with both cases : known and unknown preference directions of criteria.
Thus, it also infers the preference directions of unknown ones (`l_unk_pref_dirs`).
Tests are limited to 2 categories.

## META-SP : meta_sp.py

The file to execute this metaheuristic that infers MR-Sort models with single-peaked criteria, from assignment examples.
The algorithm assumes that the preference directions of criteria are already known.
Tests are limited to 2 categories.


# Tips

For debugging purposes, use the following piece of code which stops the execution of the code at that line (see https://docs.python.org/3/library/pdb.html):
`import pdb; pdb.set_trace()`.
In order to run a serie of tests, it is possible to construct an adhoc loop where in each iteration, a set of problem/test/algorithm parameters is defined and the global variable DATADIR is updated (for instance by incrementing the name number of the folder) in order to get organized results.

