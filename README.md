# AI-based IDS in IoT-5G networks

## Introduction

This repository presents a **Federated Learning (FL)-based Intrusion Detection System (IDS)** tailored for **5G-enabled IoT environments**. The proposed framework leverages **AI-driven deep learning models** trained collaboratively across multiple decentralized nodes. Each node trains its model locally on private data and shares only model updates with a central aggregator, eliminating the need to transmit raw data. This approach enhances privacy, reduces bandwidth usage, and complies with data protection regulations.

To further reinforce privacy guarantees, the system integrates **Differential Privacy (DP)** into the FL process. DP ensures that shared model updates cannot be reverse-engineered to reveal individual data points, protecting sensitive user information even in adversarial settings.

The IDS architecture adopts a **two-stage design**:
1. **Attack Detection (AD)**: classifies traffic as benign or malicious.
2. **Attack Classification (AC)**: identifies the specific cyberattack type.

This modular structure simplifies training, improves system interpretability, and allows fine-grained optimization for each task.

## Overview

The increasing adoption of IoT devices and the deployment of 5G networks introduce new security challenges. Centralized IDS approaches struggle with data privacy concerns and scalability. Federated Learning offers a solution by training models locally on distributed data without exposing raw information.

In this project, we:
- Process the [**CIC-IoT 2023** dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
- Simulate federated training with **three client nodes** under varying data distributions.
- Test different deel learning models, aggregation functions and training settings.
- Explore **privacy-preserving training** using DP.
- Evaluate performance under both **attack detection** and **attack classification** scenarios.


## Install dependencies

To install all the necessary Python packages listed in the `requirements.txt` file, run the following command:

```
pip install -r requirements.txt
```

## Repository structure
```
.
├── notebooks/
│   ├── ReduceDataset.ipynb
│   ├── DataPartitions.ipynb
│   ├── HyperparameterTuning.ipynb
│   ├── FLTraining.ipynb
│   ├── Validation.ipynb
│   └── Evaluation.ipynb
├── datasets/
│   ├── UNIFORM/
│   └── UNBALANCED/
├── models/                 # Saved global models after each FL round
│   ├── AD/
│      └── DP/
│   └── AC/
│      └── DP/
├── metrics/                # Performance metrics of each training setup
│   ├── AD/
│      └── DP/
│   └── AC/
│      └── DP/
├── requirements.txt
└── README.md
```

## Notebook Outline

Each Jupyter notebook encapsulates a core part of the workflow, designed to be run sequentially:

1. [`ReduceDataset.ipynb`](notebooks/ReduceDataset.ipynb): **Dataset Preparation**  
   - Loads and processes the [**CIC-IoT 2023** dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html).
   - Balances the number of samples per attack type
   - Reduces the number of benign traffic samples to roughly match the total attack volume.  
   - Outputs a compact, balanced dataset for experimentation.

2. [`DataPartitions.ipynb`](notebooks/DataPartitions.ipynb): **Federated Data Simulation**  
   - Creates two data different partition schemes of the dataset across 3 simulated client nodes:  
     - **Uniform**: All nodes receive all attack types in equal distribution, representing an ideal scenario.  
     - **Unbalanced**: Rare/underrepresented attacks are assigned to specific nodes, simulating a realistic non-IID setting typical in IoT environments.  
   - Generates:
     - A **test dataset** for global model evaluation.  
     - A **label encoder file** for class-to-index mapping.

3. [`HyperparameterTuning.ipynb`](notebooks/HyperparameterTuning.ipynb): **Model Optimization**  
   - Uses **Optuna** for automated hyperparameter tuning across different Deep Learning (DL) models.  
   - Each model has a dedicated study for reproducible and efficient search.  
   - Outputs optimal configurations for each model and task.

4. [`FLTraining.ipynb`](notebooks/FLTraining.ipynb): **Federated Model Training**  
   - Implements Federated Learning using PyTorch across 3 clients.  
   - Configurable parameters include:
     - **Mode**: Attack Detection (AD), or Attack Classification (AC).  
     - **Privacy**: With or without **Differential Privacy**.
     - **Model architecture**.
     - **Aggregation method**.
     - Number of **local epochs** and **global rounds**.
   - After each global round saves the global model in the `models/` folder.
   - Once the training is completed, logs metrics are stored in the `metrics/` folder.

5. [`Validation.ipynb`](notebooks/Validation.ipynb): **Training Analysis and Comparison**  
   - Generates comparative plots from the training metrics.
   - Facilitates comparison of configurations across models, aggregation functions, and training modes.
   - Analyzes the trade-off between privacy and utility.
   - Helps identify the most effective combination for each task.

6. [`Evaluation.ipynb`](notebooks/Evaluation.ipynb): **Model Evaluation**  
   - Evaluates the trained global models using the **test dataset**.
   - Provides performance metrics and visualizations to asses model effectiveness and generalization capabilities.


## Key Features
- **Federated Training**: Mimics decentralized learning across multiple clients without sharing raw data.
- **Attack Detection & Classification**: Supports both binary and multi-class classification security tasks.
- **Differential Privacy**: Additional integration for privacy-preserving training.
- **Flexible Experimentation**: Modular configuration for models, aggregators, and privacy controls.
- **In-depth Validation & Visualization**: Detailed metric tracking and plotting to assess performance.


## Notes

The reduced dataset, performance metrics and model checkpoints are not included due to size. Please generate them using the provided notebooks.


## References 

Finally, this project is the continuation of two studies:

- Bellmunt Fuentes, A. (2024, June 26). *Federated Transfer Learning-based Intrusion Detection System in 5G networks* (Treball Final de Grau). UPC, Facultat d'Informàtica de Barcelona, Departament d'Arquitectura de Computadors. http://hdl.handle.net/2117/416079
  
  And correspoding code can be found in the following GitHub repository: https://github.com/andreabellmunt/FTL-based-IDS-in-5G-networks<br>

- Rodríguez, E.; Valls, P.; Otero, B.; Costa, J.J.; Verdú, J.; Pajuelo, M.A.; Canal, R. *Transfer-Learning-Based Intrusion Detection Framework in IoT Networks.* Sensors 2022, 22, 5621. https://doi.org/10.3390/s22155621.
  
  An the Github repository is: https://github.com/polvalls9/Transfer-Learning-Based-Intrusion-Detection-in-5G-and-IoT-Networks<br>

## Author

Núria Arqués
