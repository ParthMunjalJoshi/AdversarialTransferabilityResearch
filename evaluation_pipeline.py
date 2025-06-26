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
    filepath = "tmp/temp.weights.h5"
    sha256_hash = hashlib.sha3_256()
    model.save_weights(filepath)
    with open(filepath, "rb") as f:
        # Read and update hash string value in chunks
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    os.remove(filepath)
    return sha256_hash.hexdigest()

#Common preprocessing functions encapsulated to reduce redundancy
def preprocess_dataset(x_test,y_test,shape,n_classes,subset_size,normalize_flag=True):
    if normalize_flag:
        x_test = x_test.astype('float32') / 255.0             #Normalize
    x_test = x_test.reshape(shape)                            #Reshape
    y_test = keras.utils.to_categorical(y_test, n_classes)    #Convert to One-Hot Encoding
    x_test = x_test[:subset_size]                             #Subset of images
    y_test = y_test[:subset_size]                             #Corresponding Subset of labels
    return (x_test,y_test)

#loads and returns test-set from MNIST dataset of given subset size
def load_mnist_testset(subset_size):
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    return (dataset_details,x_test,y_test)

#loads and returns test-set from Fashion-MNIST dataset of given subset size
def load_fmnist_testset(subset_size):
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    return (dataset_details,x_test,y_test)

##loads and returns test-set from Cifar-10 dataset of given subset size
def load_cifar10_testset(subset_size):
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,32,32,3),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(32, 32, 3),"vrange":(0.0, 1.0)}
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
    return (classifier_helper(classical_model,dataset_details), classifier_helper(hybrid_model,dataset_details))

#Generate Adversarial Attacks on model
def generate_adversarial_attacks(classifier,x_test,epsilon=0.01,maximum_iterations=40):
    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon)
    pgd = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=(2 * epsilon) / maximum_iterations, max_iter=maximum_iterations, batch_size=32, verbose=True)
    cw = CarliniL2Method(classifier=classifier, targeted=False, learning_rate=0.01, max_iter=maximum_iterations, binary_search_steps=5, confidence=0.0, initial_const=0.01)
    return (x_test,fgsm.generate(x_test),pgd.generate(x_test),cw.generate(x_test))

#Generate Min/Mean Perturbation size for model
def generate_pert(classifier,x_pgd,x_cw,x_test):
    min_perturbation_fgsm = empirical_robustness(classifier=classifier, x=x_test, attack_name="fgsm", attack_params={"eps": 0.1})
    avg_pgd_perturbation = np.mean(np.linalg.norm((x_pgd - x_test).reshape(len(x_test), -1), axis=1))
    avg_cw_perturbation = np.mean(np.linalg.norm((x_cw - x_test).reshape(len(x_test), -1), axis=1))
    return (0,min_perturbation_fgsm,avg_pgd_perturbation,avg_cw_perturbation)

#Evaluate model for a specific attack.
#Using weighted avg for precision,f1,auc-roc as well as OneVsRest strategy for auc-roc this is done to extend these to multi-class classification
def evaluate_model(classifier,x_test,y_test):
    preds = classifier.predict(x_test)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    precision = precision_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted', zero_division=0)
    f1 = f1_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted', zero_division=0)
    auc = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    return (acc,precision,f1,auc)

#Evaluate transferability for particular attack
def evaluate_transferability(classifier,x_test_cross,y_test,base_acc_atk):
    preds = classifier.predict(x_test_cross)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    TSR = 1 - acc
    accuracy_drop = base_acc_atk - acc
    return (TSR,accuracy_drop)

#Load Test set from dataset according to name of apt size
def load_testset(name,size):
    if name == 'mnist':
        return load_mnist_testset(size)
    elif name == 'fmnist':
        return load_fmnist_testset(size)
    elif name =='cifar10':
        return load_cifar10_testset(size)
    else:
        raise ValueError("Dataset Not Supported")

#Pipeline for complete evaluation of two models trained on the given dataset
def eval_pipeline(dataset_name,classical_model, hybrid_model, epsilon=0.01, maximum_iterations=40, subset_size=1000):
    #Loading given dataset's test-set
    dataset_details,x_test,y_test = load_testset(dataset_name,subset_size)
    #Generating ART Classifiers
    classifiers = create_ART_classifiers(classical_model, hybrid_model,dataset_details)
    #Measuring Robustness
    robustness_data = []
    attacks = ["clean","fgsm","pgd","cw"]
    perts = ["none","min","mean","mean"]
    hashs_set, all_adversarial_examples, all_accuracies = [], [], [] 
    for classifier in classifiers:
        hash_value = generate_sha3_256_hash(classifier.model)
        hashs_set.append(hash_value)
        adv_examples = generate_adversarial_attacks(classifier,x_test,epsilon,maximum_iterations)
        all_adversarial_examples.append(adv_examples)
        perturbations = generate_pert(classifier,adv_examples[2],adv_examples[3],x_test)
        for idx,(adv_example,perturbation) in enumerate(zip(adv_examples,perturbations)):
            metrics = evaluate_model(classifier,adv_example,y_test)
            all_accuracies.append(metrics[0])
            robustness_data.append([hash_value,attacks[idx]]+list(metrics)+[perts[idx],perturbation])
    robustness_dataframe = pd.DataFrame(robustness_data,columns=["CNN_ID","attack_type","acc","precision","f1","auc-roc","pert_type","pert_size"])
    transfer_data = []
    for i, attack in enumerate(attacks[1:]):
        metrics = evaluate_transferability(classifiers[1], all_adversarial_examples[0][i+1], y_test, all_accuracies[5+i])
        transfer_data.append([hashs_set[0], hashs_set[1], attack]+list(metrics))
        metrics2 = evaluate_transferability(classifiers[0], all_adversarial_examples[1][i+1], y_test, all_accuracies[1+i])
        transfer_data.append([hashs_set[1], hashs_set[0], attack]+list(metrics2))
    transfer_dataframe = pd.DataFrame(transfer_data,columns=["Donor_ID","Recipient_ID","attack_type","TSR","acc_drop"])
    return robustness_dataframe,transfer_dataframe
    