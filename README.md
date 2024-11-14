# Mestrado-em-Engenharia-Eletrica
Neste repositório estão armazenados somente os módulos e notebooks desenvolvidos e/ou revisados, complementando os módulos do MAIS, e um notebook de análise de agrupaemnto da Classe 1 (Experimento 6), para implementação dos experimentos relatados na dissertação de Mestrado em Engenharia Elétrica de Antonio Alberto Moreira de Azevedo. Portanto, no Pacote do MAIS, o módulo tune_lgbm.py_ foi susbstituído pelo tune_lgbm_dagshub.py (\*), contendo as funções necessárias para publicar os resultados dos experimentos e cenários associados, de treinamento e teste, inclusive os modelos treinados, no repositório MLFLOW. Foi adicionado o notebook plot_results.ipynb (\*), para emissão dos desenhos das matrizes de confusão, matrizes de confusão das anomalias agrupadas, e para os gráficos representativos de evolução temporal dos sensores e das inferências por instância usadas nos experimentos (alarme de evento), com legendas no idioma português. Deve ser substituído o label_mappers.py (\*) pelo arquivo deste repositório, pois foram adicionadas classes com objetivo de realizar os Experimentos 1 e 2 da dissertação. O LEIA-ME do MAIS foi revisado (verão 2.0),conforme disposto abaixo, constando as alterações mencionadas acima. Os demais módulos do MAIS não foram alterados e podem ser carregados no repositório https://github.com/petrobras/3W/tree/main/toolkit/mais.

This repository stores only the modules and notebooks developed and/or revised, complementing the MAIS modules, and a Class 1 cluster analysis notebook (Experiment 6), for the implementation of the experiments reported in the Master's dissertation in Electrical Engineering by Antonio Alberto Moreira de Azevedo. Therefore, in MAIS package the tune_lgbm.py module was replaced by tune_lgbm_dagshub.py (\*), containing the functions necessary to publish the results of the experiments and associated training and testing scenarios, including the trained models, in the MLFLOW repository. The plot_results.ipynb (\*) notebook was added, to issue the drawings of the confusion matrices, confusion matrices of the grouped anomalies, and for the representative graphs of the temporal evolution of the sensors and the inferences per instance used in the experiments (event alarm), with captions in Portuguese. label_mappers.py (\*) must be replaced with the file from this repository, considering that classes were added to it to perform Experiments 1 and 2 of dissertation. The MAIS README has been revised (version 2.0), as set out below, including the changes mentioned above. The other MAIS modules have not been changed and can be uploaded to the repository https://github.com/petrobras/3W/tree/main/toolkit/mais.


# Modular Artificial Intelligence System (MAIS), version 2.0
(New version of README.md of MAIS)

This repository presents MAIS, a system that implements Machine Learning techniques on a modular way, enabling the developer to test his/her own experiments and/or adapting others esperimentsexperiments with their own idea. MAIS was developed by the Signal, Multimedia and Telecommunications (SMT) laboratory with the help from Petrobras.

In this version, MAIS implements a multiclass LGBM classifier, with the following optional features:

* Statistical features;
  * Regular average withnwithin an window;
  * ExponetiallyExponentially weigthedweighted average within ana window;.
* Wavelets features;
* Imputation methods: keep NaN values, change by the mean, ...
* Different labeling methods;
  * Using the most recent label from ana window as the laellabel for that sample; or
  c* Using the label in the middle of ana window as the laellabel for that sample.
* Feature selection using Random Forest importance

# Repository Structure

```
├── environment.yml
├── training
│   └── multiclass
│       ├── experiments
│       │   ├── base_experiment.py
│       │   ├── multi_ew_stats_mrl_nonan.py
│       │   ├── ...
|       |   └── plot_results.ipynb *
│       └── tune_lgbm_dagshub.py *
├── dataset
│   └── dataset.py
├── processing
│   ├── feature_mappers.py
│   └── label_mappers.py *
├── utils
│   └── utils.py
├── visualization
│   └── generate_report.py
└── setup.py
```
MAIS uses a class called Experiment, which contains all the necessary steps to create your experiment.  So, under the folder "experimentstraining/multiclass/experiments/", you add your custom Experiment class, based on the BaseExperiment,  defined on "experimentstraining/multiclass/experiments/base_experiment.py". Some examples are already implemented in order toto give an idea on how an experiment is created.

The "mais/" folder contains classes definitions that create everything that is used to create an experiment, i.e., contains all utility classes . Some of them are 

  1. mais/dataset/dataset.py: Defines the class MAEDataset, which contains the core logic behind MAIS dataloader.  Some of its functions are: read a .csv, read the feature extraction, create the final table (the model input).
  2. mais/dataprocessing/feature\_mappers.py: Defines the classes that extract the attributes for a given experiment. t The implementation uses torch in order toto make the extraction faster when using a lot of data. In the current version there are some strategies already implemented, for example: 
     1. TorchStatisticalFeatureMapper: created statistical features (9/tag) from a rectangular window;
     2. TorchWaveletFeatureMapper: creates wavelets features;
     3. TorchEWStatisticalFeatureMapper: creates statistical features from a window with exponential weights for each sample.
  3. mais/dataprocessing/label\_mappers.py: Creates the classes that define how the detection is done. For example, it is possible to choose if the transient period of signal will be considered, or if the samples in the beggining of a file (which are usually not faulty) will be considered.
4. mais/training/tune\_lgbm\_dagshub.py: Initializes a mlflow experiment containing all runs from the Bayesian optimization search.  For every run, the script trains a model and saves its metrics.  It was developed for PYTHON.  If you intend to work on WINDOWS, you must comment in line 483 [os.nice(19)].  The os.nice() method is only available on UNIX platforms. This module is responsible for the multiclass classifier. Specific functions were created in it to publish the results of the experiments and associated training and testing scenarios, including the trained models, in the MLFLOW repository.

  So, to add new utility functions and/or classes, the "mais/" folder is probably the best place (under the correspondent file). For example, if one needs to create a new feature extractor, the best way to proceed is creating a new FeatureMapper under the file  "mais/data/feature\_mappers.py".

# Experiment examples

In the folder experiments/multiclass/ there are many examples that can guide on how to create a new one:
1. multi_ew_stats_mrl_nonan.py
2. multi_mixed_mrl_nonan.py
3. multi_mixed_select_mrl_nonan.py
4. multi_stats_mrl_nonan.py
5. multi_stats_select_mrl_nonan.py
6. multi_wavelets_mrl_nonan.py
7. multi_wavelets_select_mrl_nonan.py

The name of theses experiments reflect what they implements, for examples, the experiment "multi_stats_select_mrl_nonan.py" implements a multiclass classifier that uses statistical features, a feature selector, uses the most recent label as the label associated to a window and imputes NaN values. The acronym we used are:

* multi = Multiclass Experiment;
* ew = Exponentially weighted;
* stats = Statistical features;
* mrl = Most recent label;
* nonan = NaN imputation;
* mixed = both statistical and wavelet features; 
* select = Feature selector (Random forest feature selection); and
* wavelets = Wavelets

For example, between experiments 1 and 6, the difference is the kind of features will be computed. In the first on the exponentially weighted statistical features are used and in the second one, just the wavelets. And to do that, the difference is basically assign the correspondent feature wrapper. 

# Using functions outside MAIS package

The functions defined within MAIS package are being exposed to mais package main folder. So, from the 3W folder one can use the RollingLabelStrategy class for example by just doing:

`from toolkit.mais import RollingLabelStrategy`.

In a future, it is possible to make it transparent to the user, abstracting the MAIS package path. For that, two steps are necessary:

1. Add the proper import into __init__.py file in 3W package; and
2. Double check if there are any functions with the same name.

# How to use
It is recommended that you organize your package with the same structure (expressed in terms of a hierarchical filesystem).  The __init__.py files are required to make Python treat directories containing the file as packages. This prevents directories with a common name, such as string, from unintentionally hiding valid modules that occur later on the module search path. In the simplest case, __init__.py can just be an empty file, but it also executes initialization code for the package, In the mais subdirectory.
Also, In the mais subdirectory you shall download the setup.py which contains the instructions for setuptools to install the package.
Outside mais package, in module 3W/toolkit/base.py, you shall insert your path of PATH_3W_PROJECT (line 15), and you must make shure that dataset.ini is in dataset fold, in your environment.

After, you can creating the experiment and putting it into the experiment folder (for example, 'experiments/multiclass/experiments/example.py',. 
Once your basic package architecture is built, you can install it using pip. To install your package, make sure you are in the 3W directory of environment.

To run tune_lgbm_dagshub.py:

Install the 3W libraries and MAIS, via terminal:

- Go to the 3W directory and use "pip install -e toolkit";
  
- Go to the toolkit directory and use "pip install -e mais".


  1. [OPTIONAL] Initialize a mlflow server with an sqlite URI for the logs.  
  2. Execute 'tune\_lgbm\_dagashub.py'. This script initializes a mlflow experiment containing all runs from the Bayesian optimization search.  For every run, the script trains a model and saves its metrics.  The commands to execute 'tune\_lgbm.py' are:
     a) data-root: Root directory with the data;
     
     b) experiment-name: Name of the experiment (must be inside 'experiments/multiclass/experiments/');
     
     c) num-trials: Number of Bayesian optimization trials;
     
     d) inner-splits: Number of cross-validation inner loops;
     
     e) outer-splits: Number of cross-validation outer loops;
     
     f) n-jobs: Number of cores available for parallel processing;
     
As example, one may execute in terminal: python “os.environ[tune_lgbm]”/tune_lgbm_dagshub.py tune -t “os.environ['training dataset']”/ 'training dataset name’  -T os.environ['test dataset']/ 'test dataset name’ -e “experiment name” -n 100

All this these commands can be also consulted using --help. [P.S: Use appropriate environment variables for your mlflow log system.]
