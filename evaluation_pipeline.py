"""
The aim of this pipeline is to streamline the process of evaluating adversarial robustness and transferability
between a pair of models by extracting key metrics and storing them in a dataframe.
"""
#Dependancies
import tensorflow as tf
from tensorflow import keras 
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method
from sklearn.metrics import precision_score, f1_score, roc_auc_score
from art.metrics import empirical_robustness
import numpy as np
import pandas as pd
import hashlib
import os

#Generates hash for given model
def generate_sha3_256_hash(model):
    """
    Generates a sha3 256bit hash of model to serve as ID.
    """
    filepath = "tmp/temp.weights.h5"
    sha256_hash = hashlib.sha3_256()
    model.save(filepath)
    with open(filepath, "rb") as f:
        # Read and update hash string value in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    try:
        os.remove(filepath)
    except Exception:
        raise PermissionError
    return sha256_hash.hexdigest()

#Common preprocessing functions encapsulated to reduce redundancy
def preprocess_dataset(x_test,y_test,shape,n_classes,subset_size,normalize_flag=True):
    """
    Normalizes, Resizes images and converts labels to one-hot encoding. Clips dataset to subset size.
    """
    if normalize_flag:
        x_test = x_test.astype('float32') / 255.0             #Normalize
    x_test = x_test.reshape(shape)                            #Reshape
    y_test = keras.utils.to_categorical(y_test, n_classes)    #Convert to One-Hot Encoding
    x_test = x_test[:subset_size]                             #Subset of images
    y_test = y_test[:subset_size]                             #Corresponding Subset of labels
    return (x_test,y_test)

#loads and returns test-set from MNIST dataset of given subset size
def load_mnist_testset(subset_size):
    """
    Loading the MNIST dataset.
    """
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {
        "n_classes":10,
        "shape":(28, 28, 1),
        "vrange":(0.0, 1.0),
    }
    return (dataset_details,x_test,y_test)

#loads and returns test-set from Fashion-MNIST dataset of given subset size
def load_fmnist_testset(subset_size):
    """
    Loading the Fashion-MNIST dataset.
    """
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {
        "n_classes":10,
        "shape":(28, 28, 1),
        "vrange":(0.0, 1.0),
    }
    return (dataset_details,x_test,y_test)

#Helper Function to create ART Classifiers
def classifier_helper(model,dataset_details):
    classifier = TensorFlowV2Classifier(
        model=model,
        loss_object=tf.keras.losses.CategoricalCrossentropy(),
        nb_classes=dataset_details["n_classes"], #10
        input_shape=dataset_details["shape"],    #(28, 28, 1)
        clip_values=dataset_details["vrange"],   #(0.0, 1.0)
    )
    return classifier

#returns classifier for given models
def create_ART_classifiers(classical_model, hybrid_model,dataset_details):
    """
    Creates ART classifers for given Tensorflow+keras models.
    """
    classical_classifier,hybrid_classifier = classifier_helper(classical_model,dataset_details), classifier_helper(hybrid_model,dataset_details)
    return (classical_classifier,hybrid_classifier)

#Generate 3-Adversarial Attacks on model
def generate_adversarial_attacks(classifier,x_test,epsilon=0.01,maximum_iterations=40):
    """
    Generates adversarial test samples on x_test wrt. classifier for FGSM,PGD and CW attacks.
    """
    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_fgsm = fgsm.generate(x=x_test)
    pgd = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=(2 * epsilon) / maximum_iterations, max_iter=maximum_iterations, batch_size=32, verbose=True)
    x_pgd = pgd.generate(x=x_test)
    classical_cw = CarliniL2Method(classifier=classifier, targeted=False, learning_rate=0.01, max_iter=maximum_iterations, binary_search_steps=5, confidence=0.0, initial_const=0.01)
    x_cw = classical_cw.generate(x=x_test)
    return (x_fgsm,x_pgd,x_cw)

#Generate Min/Mean Perturbation size for model
def generate_pert(classifier,x_pgd,x_cw,x_test):
    """
    Generates certified minimum perturbation size for FGSM attack (non-iterative) and Average perturbation size for iterative attacks
    We can fine-tune the attack values using these details.
    """
    min_perturbation_fgsm = empirical_robustness(classifier=classifier, x=x_test, attack_name="fgsm", attack_params={"eps": 0.1})
    pgd_perturbations = np.linalg.norm((x_pgd - x_test).reshape(len(x_test), -1), axis=1)
    avg_pgd_perturbation = np.mean(pgd_perturbations)
    cw_perturbations = np.linalg.norm((x_cw - x_test).reshape(len(x_test), -1), axis=1)
    avg_cw_perturbation = np.mean(cw_perturbations)
    return (min_perturbation_fgsm,avg_pgd_perturbation,avg_cw_perturbation)

#Evaluate model for a specific attack
#Using weighted avg for precision,f1,auc-roc as well as OneVsRest strategy for auc-roc this is done to extend these to multi-class classification
def evaluate_model(classifier,x_test,y_test):
    """
    Evaluates model robustness and returns acc,precision,f1-score,auc-roc for a specified adversarial set
    precision,f1-score and auc-roc use weighted average for conversion to multi-class classification metric
    auc-roc uses OvR (One vs Rest) strategyto break multi-class scenario to binary class analogue.
    """
    preds = classifier.predict(x_test)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    precision = precision_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted')
    f1 = f1_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted')
    auc = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    return (acc,precision,f1,auc)

#Evaluate transferability for particular attack
def evaluate_transferability(classifier,x_test_cross,y_test,base_acc_atk):
    """
    Evaluates adversarial transferability and returns Transfer Success Rate and Accuracy Drop.
    """
    preds = classifier.predict(x_test_cross)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    TSR = 1 - acc
    accuracy_drop = base_acc_atk - acc
    return (TSR,accuracy_drop)

#Load Test set from dataset according to name of apt size
def load_testset(name,size):
    """
    calls helper fxn based on name of attack.
    """
    if name == 'mnist':
        return load_mnist_testset(size)
    elif name == 'fmnist':
        return load_fmnist_testset(size)
    else:
        print("Dataset Not Supported")
        raise ValueError

#Pipeline for complete evaluation of two models trained on the given dataset
def eval_pipeline(dataset_name,classical_model, hybrid_model, epsilon=0.01, maximum_iterations=40, subset_size=1000):
    #Loading given dataset's test-set
    dataset_details,x_test,y_test = load_testset(dataset_name,subset_size)
    #Generating ART Classifiers
    classical_classifier,hybrid_classifier = create_ART_classifiers(classical_model, hybrid_model,dataset_details)
    #Generating Adversarial Examples
    classical_x_fgsm,classical_x_pgd,classical_x_cw = generate_adversarial_attacks(classical_classifier,x_test,epsilon,maximum_iterations)
    hybrid_x_fgsm,hybrid_x_pgd,hybrid_x_cw = generate_adversarial_attacks(hybrid_classifier,x_test,epsilon,maximum_iterations)
    #Generating Perturbation sizes
    classical_min_pert_fgsm, classical_avg_pert_pgd,classical_avg_pert_cw = generate_pert(classical_classifier,classical_x_pgd,classical_x_cw,x_test)
    hybrid_min_pert_fgsm, hybrid_avg_pert_pgd,hybrid_avg_pert_cw = generate_pert(hybrid_classifier,hybrid_x_pgd,hybrid_x_cw,x_test)
    #Measuring clean robustness metrics
    classical_acc_clean,classical_precision_clean,classical_f1_clean,classical_auc_clean = evaluate_model(classical_classifier,x_test,y_test)
    hybrid_acc_clean,hybrid_precision_clean,hybrid_f1_clean,hybrid_auc_clean = evaluate_model(hybrid_classifier,x_test,y_test)
    #Measuring fgsm robustness metrics
    classical_acc_fgsm,classical_precision_fgsm,classical_f1_fgsm,classical_auc_fgsm = evaluate_model(classical_classifier,classical_x_fgsm,y_test)
    hybrid_acc_fgsm,hybrid_precision_fgsm,hybrid_f1_fgsm,hybrid_auc_fgsm = evaluate_model(hybrid_classifier,hybrid_x_fgsm,y_test)
    #Measuring pgd robustness metrics
    classical_acc_pgd,classical_precision_pgd,classical_f1_pgd,classical_auc_pgd = evaluate_model(classical_classifier,classical_x_pgd,y_test)
    hybrid_acc_pgd,hybrid_precision_pgd,hybrid_f1_pgd,hybrid_auc_pgd = evaluate_model(hybrid_classifier,hybrid_x_pgd,y_test)
    #Measuring Carlini-Wagner robustness metrics
    classical_acc_cw,classical_precision_cw,classical_f1_cw,classical_auc_cw = evaluate_model(classical_classifier,classical_x_cw,y_test)
    hybrid_acc_cw,hybrid_precision_cw,hybrid_f1_cw,hybrid_auc_cw = evaluate_model(hybrid_classifier,hybrid_x_cw,y_test)

    #Measuring Classical -> Hybrid Transferability for FGSM attack
    ch_fgsm_tsr, ch_fgsm_acc_drop = evaluate_transferability(hybrid_classifier,classical_x_fgsm,y_test,hybrid_acc_fgsm)
    #Measuring Classical -> Hybrid Transferability for PGD attack
    ch_pgd_tsr, ch_pgd_acc_drop = evaluate_transferability(hybrid_classifier,classical_x_pgd,y_test,hybrid_acc_pgd)
    #Measuring Classical -> Hybrid Transferability for CW attack
    ch_cw_tsr, ch_cw_acc_drop = evaluate_transferability(hybrid_classifier,classical_x_cw,y_test,hybrid_acc_cw)

    #Measuring Hybrid -> Classical Transferability for FGSM attack
    hc_fgsm_tsr, hc_fgsm_acc_drop = evaluate_transferability(classical_classifier,hybrid_x_fgsm,y_test,classical_acc_fgsm)
    #Measuring Hybrid -> Classical Transferability for PGD attack
    hc_pgd_tsr, hc_pgd_acc_drop = evaluate_transferability(classical_classifier,hybrid_x_pgd,y_test,classical_acc_pgd)
    #Measuring Hybrid -> Classical Transferability for CW attack
    hc_cw_tsr, hc_cw_acc_drop = evaluate_transferability(classical_classifier,hybrid_x_cw,y_test,classical_acc_cw)
    #Creating Dataframe for Robustness Data
    classical_hash = generate_sha3_256_hash(classical_model)
    hybrid_hash = generate_sha3_256_hash(hybrid_model)
    robustness_data = {
        "CNN_ID": 4*[classical_hash]+4*[hybrid_hash],
        "attack_type": ['clean','fgsm','pgd','cw']*2,
        "accuracy":[classical_acc_clean,classical_acc_fgsm,classical_acc_pgd,classical_acc_cw,hybrid_acc_clean,hybrid_acc_fgsm,hybrid_acc_pgd,hybrid_acc_cw],
        "precision":[classical_precision_clean,classical_precision_fgsm,classical_precision_pgd,classical_precision_cw,hybrid_precision_clean,hybrid_precision_fgsm,hybrid_precision_pgd,hybrid_precision_cw],
        "f1_score":[classical_f1_clean,classical_f1_fgsm,classical_f1_pgd,classical_f1_cw,hybrid_f1_clean,hybrid_f1_fgsm,hybrid_f1_pgd,hybrid_f1_cw],
        "auc_roc":[classical_auc_clean,classical_auc_fgsm,classical_auc_pgd,classical_auc_cw,hybrid_auc_clean,hybrid_auc_fgsm,hybrid_auc_pgd,hybrid_auc_cw],
        "perturbation_type":['null','min','avg','avg']*2,
        "perturbation_size":[0.0,classical_min_pert_fgsm,classical_avg_pert_pgd,classical_avg_pert_cw,0.0,hybrid_min_pert_fgsm,hybrid_avg_pert_pgd,hybrid_avg_pert_cw]
    }
    robustness_dataframe = pd.DataFrame(robustness_data)
    #Creating Dataframe for Transferability Data
    transferability_data = {
        "CNN_Donor_ID":[classical_hash]*3 + [hybrid_hash]*3,
        "CNN_Recipient_ID":[hybrid_hash]*3 + [classical_hash]*3,
        "attack_type":['fgsm','pgd','cw']*2,
        "TSR":[ch_fgsm_tsr,ch_pgd_tsr,ch_cw_tsr,hc_fgsm_tsr,hc_pgd_tsr,hc_cw_tsr],
        "acc_drop":[ch_fgsm_acc_drop,ch_pgd_acc_drop,ch_cw_acc_drop,hc_fgsm_acc_drop,hc_pgd_acc_drop,hc_cw_acc_drop]
    }
    transferability_dataframe = pd.DataFrame(transferability_data)
    return (robustness_dataframe,transferability_dataframe)
