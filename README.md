# AdversarialTransferabilityResearch

This project investigates the **adversarial transferability** of gradient-based attacks, specifically **Fast Gradient Sign Method (FGSM)**, **Projected Gradient Descent (PGD)**, and **Carlini-Wagner (CW)**, between Classical Convolutional Neural Networks (CNNs) and Hybrid Quantum-Classical CNNs (HQCNNs). This research focuses on the entanglement strategies used in HQCNNs.

 
## Project Overview

Adversarial examples, crafted on one type of model (classical or hybrid), are evaluated on the other to analyze their **cross-model vulnerability**. This bidirectional assessment aims to reveal the extent to which adversarial perturbations generated on one architecture can affect the other.

## Why is this important?

Understanding adversarial transferability is crucial for several reasons:

  * **Grey-Box and Black-Box Attacks:** In real-world scenarios, attackers often don't have full access to a model's architecture or parameters. Transferability allows adversaries to exploit shared weaknesses even with limited information. This is done by training adversarial examples on a surrogate model and using them to attack the target model.
  * **Robust AI Systems:** The findings from this study will inform the design of more robust and secure AI systems, particularly within the evolving landscape of quantum machine learning. It will deliver insight as to which entanglement strategies are more adversarially robust and less-prone to transfer attacks.

## Methodology

![plot](./lib/imgs/EntanglementExperiment.png)

The experiment shall generate lists describing entanglement layers in a defined model architecture.
This shall be passed to subsequent modules for model creation, model training and evaluation.

### Model Creation:

![plot](./lib/imgs/EntglModelFactory.png)

This module creates a HQCNN/Classical model based on parameters passed to it. For exact specifics refer to the files:

  * entanglement\_model\_factory.py
  * \_entanglement\_circuit.py
  * \_entanglement\_layer.py

### Model Training:

![plot](./lib/imgs/TrainModule.png)

This module takes a model and a dataset and trains the model on that specific dataset. Training data is split 9:1 into training
and validation data. Model is trained using the Adam Optimizer. For exact specifics refer to the file:

  * training\_module.py

### Model Evaluation:

![plot](./lib/imgs/EvalPipe.png)

This pipeline takes two models and calculates adversarial robustness and transferability metrics for them and returns dataframes.

:warning: **CAUTION** :warning: By default Carlini-Wagner attacks are disabled in the pipeline as they are computationally 
expensive. If you wish to include them you can do so by setting carlini_wagner_flag = True in evaluation_pipeline.py
before running the experiment.

The structure of the dataframes is given by:

![plot](./lib/imgs/Dataset_design.png)

In transferability metrics, TSR stands for "transfer success rate" and is the fraction of adversarial examples that successfully fool the model. The accuracy gap measures the convergence between self- and transfer attacks. A lower accuracy gap implies a transfer attack was as effective as a white-box attack, which is the theoretical best-case scenario.

For exact specifics refer to the file:

  * evaluation\_pipeline.py

### Performance metrics:
This project yielded significant improvements in experimental efficiency and throughput, driven by robust engineering practices and a focus on automation. The key performance metrics achieved are:
  * **Reduced manual overhead in evaluation by 75%** :
  Without the evaluation pipeline, robustness and bi-directional transferability would need to be assessed separately, increasing complexity and manual effort. The pipeline consolidates these evaluations into a single, streamlined step. The resulting reduction in overhead is quantified by the decrease in wait states—by minimizing idle time, the pipeline can operate in the background, enabling researchers to remain productive and continue other tasks concurrently.
  * **Up to 88% Faster Repeat Evaluations** :
  This acceleration was enabled by integrating model fingerprinting and memoization into the evaluation pipeline. When an evaluation for a specific model configuration and attack scenario had been performed previously, the pipeline now reuses the stored adversarial examples instead of re-computing them. This was measured by comparing the execution time of repeated evaluations with and without the memoization system. This improvement is particularly valuable in scenarios where only one of the two compared models is repeated, which occurs frequently.
  * **Boosted Experimental Throughput by 50%** : 
  This improvement was a direct result of automating the entire experimental workflow, including model generation, training, evaluation, and result logging. By reducing manual intervention and speeding up repeated evaluations, the total number of experiments that could be executed and analyzed within a given timeframe increased significantly. This was measured by comparing the number of completed experiment runs per unit of time before and after the automation of these processes for a small scale test experiment. 

### Datasets used:
  * MNIST dataset
  * FashionMNIST dataset
  * CIFAR - 10 dataset

### Final Output:
Running the experiment produces 3 useful csv files:
  * lib/index.csv:  contains model ID (hash) and architecture (entanglement strategy details)
  * lib/robustness.csv:  contains compiled robustness data
  * lib/transferability.csv: contains compiled transferability data

---

## Getting Started

### Prerequisites:

  * Python 3.12
  * pip >= 25.1.1

### Installation:

```bash
git clone https://github.com/ParthMunjalJoshi/AdversarialTransferabilityResearch.git
cd AdversarialTransferabilityResearch
pip install -r requirements.txt
```

### Usage:
```bash
python entanglement_experiment_homogenous.py
```

To modify configurations related to different aspects of the pipeline you may access :
  * lib/expt_config.json
 
## Expected Outcomes
The project aims to provide insights into:
  * The degree of adversarial transferability between classical and hybrid quantum-classical CNNs.
  * The impact of different entanglement strategies on adversarial robustness and transferability.

---


## Entanglement Strategies

Entanglement strategies are an important aspect of Parametric Quantum Circuit design. 

### Homogenous Strategies:

These are Ansatz using only one type of entanglement layer throughout the circuit. The following are a few types:
  * No entanglement:
    In this strategy, the PQC consists only of parametric phase gates and no entangling layers. 

    ![plot](./lib/imgs/strat/no_entg.png)


  * Linear Entanglement:
    In this strategy, we apply CNOT/CZ between neighboring qubits (N,N+1) pairs.

    ![plot](./lib/imgs/strat/linear_entg.png)

  * Circular Entanglement:
    This is an extension to the linear entanglement strategy where the last qubit is linked with the first to complete the ring.

    ![plot](./lib/imgs/strat/circular_entg.png)    

  * Full Entanglement:
    In this strategy, there is a CNOT/CZ between every pair of qubits.
  
    ![plot](./lib/imgs/strat/full_entg.png)   

  * Staggered Entanglement:
    In this strategy, linking is done in layers or stages, typically by applying entangling gates to different non-overlapping, sets of qubits in sequential blocks of the circuit.

    ![plot](./lib/imgs/strat/stag_entg.png) 

Note: in the above illustrations we have used CNOT but CZ can also produce an entangling effect.



