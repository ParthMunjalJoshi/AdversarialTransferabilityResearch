#Change this to true if you have the computational resources to carry out cw attacks
carlini_wagner_flag = False 
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
import pickle

memo_file_path = 'lib/memo.pkl'

#Load db from pickle_file
def load_db():
    """Loads the memoization database from disk.

    Returns:
        Tuple[str, dict]: Path to the memo file and the database dictionary.
    """
    db = {}
    if os.path.exists(memo_file_path):
        with open(memo_file_path, 'rb') as dbfile:
            db = pickle.load(dbfile)
    else:
        os.makedirs(os.path.dirname(memo_file_path), exist_ok=True)
    return (memo_file_path,db)

#Checks if model is already evaluated
def check_redundancy(thash,dataset):
    """Checks whether a given model hash is already evaluated.

    Args:
        thash (str): SHA3-256 hash of the model weights.

    Returns:
        bool: True if the model has already been evaluated and cached.
    """
    _,db = load_db()
    return ((dataset,thash) in db)

#Pulls model adversarial examples from storage
def load_from_memo(hash_value,dataset):
    """Loads adversarial examples for a model from the memoization database.

    Args:
        hash_value (str): SHA3-256 hash of the model.

    Returns:
        Tuple[np.ndarray, ...]: Tuple of adversarial example arrays.
    """
    _,db = load_db()
    return db[(dataset,hash_value)]
    

#Stores adversarial examples of model in a dump file
def store_adv_examples(dataset,hash_value,adv_examples):
    """Stores adversarial examples in the memoization database.

    Args:
        hash_value (str): SHA3-256 hash of the model.
        adv_examples (Tuple[np.ndarray, ...]): Generated adversarial examples to store.
    """
    memo_file_path,db = load_db()
    db[(dataset,hash_value)] = adv_examples   
    with open(memo_file_path, 'wb') as dbfile:
        pickle.dump(db, dbfile)        


#Generates hash for given model
def generate_sha3_256_hash(model):
    """Generates a SHA3-256 hash based on the weights of a TensorFlow model.

    Args:
        model (tf.keras.Model): The model to hash.

    Returns:
        str: SHA3-256 hexadecimal hash string of the model's weights.
    """
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
    """Preprocesses a dataset: normalizes, reshapes, one-hot encodes, and subsets.

    Args:
        x_test (np.ndarray): Input test images.
        y_test (np.ndarray): Corresponding labels.
        shape (Tuple[int]): Desired shape after reshaping.
        n_classes (int): Number of output classes.
        subset_size (int): Number of samples to use from the dataset.
        normalize_flag (bool, optional): Whether to normalize pixel values. Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Preprocessed inputs and labels.
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
    """Loads and preprocesses the MNIST test set.

    Args:
        subset_size (int): Number of samples to use from the test set.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]: Dataset metadata, inputs, and labels.
    """
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    return (dataset_details,x_test,y_test)

#loads and returns test-set from Fashion-MNIST dataset of given subset size
def load_fmnist_testset(subset_size):
    """Loads and preprocesses the Fashion-MNIST test set.

    Args:
        subset_size (int): Number of samples to use from the test set.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]: Dataset metadata, inputs, and labels.
    """
    (_, _), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,28,28,1),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(28, 28, 1),"vrange":(0.0, 1.0)}
    return (dataset_details,x_test,y_test)

#loads and returns test-set from Cifar-10 dataset of given subset size
def load_cifar10_testset(subset_size):
    """Loads and preprocesses the CIFAR-10 test set.

    Args:
        subset_size (int): Number of samples to use from the test set.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]: Dataset metadata, inputs, and labels.
    """
    (_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_test,y_test = preprocess_dataset(x_test,y_test,(-1,32,32,3),10,subset_size,True)
    dataset_details = {"n_classes":10,"shape":(32, 32, 3),"vrange":(0.0, 1.0)}
    return (dataset_details,x_test,y_test)

#Helper Function to create ART Classifiers
def classifier_helper(model,dataset_details):
    """Creates an ART TensorFlowV2Classifier from a Keras model and dataset metadata.

    Args:
        model (tf.keras.Model): The model to wrap.
        dataset_details (dict): Metadata including number of classes, input shape, and clipping range.

    Returns:
        TensorFlowV2Classifier: Wrapped ART classifier.
    """
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
    """Creates ART classifiers for both classical and hybrid models.

    Args:
        classical_model (tf.keras.Model): Classical neural network model.
        hybrid_model (tf.keras.Model): Quantum-enhanced (hybrid) model.
        dataset_details (dict): Dataset metadata.

    Returns:
        Tuple[TensorFlowV2Classifier, TensorFlowV2Classifier]: ART classifiers.
    """
    return (classifier_helper(classical_model,dataset_details), classifier_helper(hybrid_model,dataset_details))

#Generate Adversarial Attacks on model
def generate_adversarial_attacks(classifier,x_test,epsilon=0.01,maximum_iterations=40):
    """Generates adversarial examples using FGSM, PGD, and Carlini-Wagner attacks.

    Args:
        classifier (TensorFlowV2Classifier): ART classifier for the model.
        x_test (np.ndarray): Test inputs to attack.
        epsilon (float): Perturbation magnitude.
        maximum_iterations (int): Maximum number of iterations for iterative attacks.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Clean, FGSM, PGD, and CW adversarial examples.
    """
    fgsm = FastGradientMethod(estimator=classifier, eps=epsilon)
    pgd = ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=(2 * epsilon)/maximum_iterations, max_iter=maximum_iterations, batch_size=32, verbose=True)
    if carlini_wagner_flag:
        cw = CarliniL2Method(classifier=classifier, targeted=False, learning_rate=0.01, max_iter=maximum_iterations, binary_search_steps=5, confidence=0.0, initial_const=0.01)
        return (x_test,fgsm.generate(x_test),pgd.generate(x_test),cw.generate(x_test))
    else:
        return(x_test,fgsm.generate(x_test),pgd.generate(x_test))

#Generate Min/Mean Perturbation size for model
def generate_pert(classifier,x_pgd,x_test,x_cw=None):
    """Computes empirical robustness and average perturbation sizes.

    Args:
        classifier (TensorFlowV2Classifier): ART classifier.
        x_pgd (np.ndarray): PGD adversarial examples.
        #x_cw (np.ndarray): CW adversarial examples.
        x_test (np.ndarray): Original inputs.

    Returns:
        Tuple[float, float, float, float]: Dummy (0), empirical robustness for FGSM, PGD perturbation, CW perturbation.
    """
    min_perturbation_fgsm = empirical_robustness(classifier=classifier, x=x_test, attack_name="fgsm", attack_params={"eps": 0.1})
    avg_pgd_perturbation = np.mean(np.linalg.norm((x_pgd - x_test).reshape(len(x_test), -1), axis=1))
    if carlini_wagner_flag:
        avg_cw_perturbation = np.mean(np.linalg.norm((x_cw - x_test).reshape(len(x_test), -1), axis=1))
        return (0,min_perturbation_fgsm,avg_pgd_perturbation,avg_cw_perturbation)
    else:
        return (0, min_perturbation_fgsm, avg_pgd_perturbation)

#Evaluate model for a specific attack.
#Using weighted avg for precision,f1,auc-roc as well as OneVsRest strategy for auc-roc this is done to extend these to multi-class classification
def evaluate_model(classifier,x_test,y_test):
    """Evaluates model performance on given test set.

    Args:
        classifier (TensorFlowV2Classifier): ART classifier.
        x_test (np.ndarray): Test inputs.
        y_test (np.ndarray): True one-hot labels.

    Returns:
        Tuple[float, float, float, float]: Accuracy, weighted precision, F1 score, AUC-ROC (One-vs-Rest).
    """
    preds = classifier.predict(x_test)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    precision = precision_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted', zero_division=0)
    f1 = f1_score(np.argmax(y_test,axis=1), np.argmax(preds, axis=1), average='weighted', zero_division=0)
    auc = roc_auc_score(y_test, preds, multi_class='ovr', average='weighted')
    return (acc,precision,f1,auc)

#Evaluate transferability for particular attack
def evaluate_transferability(classifier,x_test_cross,y_test,base_acc_atk):
    """Evaluates attack transferability to the given model.

    Args:
        classifier (TensorFlowV2Classifier): Target classifier receiving transferred attack.
        x_test_cross (np.ndarray): Adversarial examples generated from another model.
        y_test (np.ndarray): True one-hot labels.
        base_acc_atk (float): Accuracy of this model on its own attack.

    Returns:
        Tuple[float, float]: Transfer Success Rate (TSR) and accuracy drop.
    """
    preds = classifier.predict(x_test_cross)
    acc = np.mean(np.argmax(preds, axis=1) == np.argmax(y_test,axis=1))
    TSR = 1 - acc
    accuracy_drop = base_acc_atk - acc
    return (TSR,accuracy_drop)

#Load Test set from dataset according to name of apt size
def load_testset(name,size):
    """Loads and returns a preprocessed test set for the specified dataset.

    Args:
        name (str): Name of the dataset ('mnist', 'fmnist', or 'cifar10').
        size (int): Number of test samples to return.

    Returns:
        Tuple[dict, np.ndarray, np.ndarray]: Dataset metadata, test inputs, and labels.

    Raises:
        ValueError: If an unsupported dataset name is provided.
    """
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
    """Performs adversarial robustness and transferability evaluation between two models.

    Evaluates classical and hybrid models against clean and adversarial examples (FGSM, PGD, CW),
    computes performance metrics, stores adversarial examples for reuse, and measures transferability
    of attacks across models.

    Args:
        dataset_name (str): Dataset name ('mnist', 'fmnist', 'cifar10').
        classical_model (tf.keras.Model): Classical model to evaluate.
        hybrid_model (tf.keras.Model): Hybrid (quantum-enhanced) model to evaluate.
        epsilon (float, optional): Perturbation magnitude for attacks. Defaults to 0.01.
        maximum_iterations (int, optional): Max iterations for PGD and CW. Defaults to 40.
        subset_size (int, optional): Number of test samples to evaluate on. Defaults to 1000.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - Robustness metrics dataframe (acc, f1, precision, auc, perturbation).
            - Transferability metrics dataframe (TSR, accuracy drop).
    """
    #Loading given dataset's test-set
    dataset_details,x_test,y_test = load_testset(dataset_name,subset_size)
    #Generating ART Classifiers
    classifiers = create_ART_classifiers(classical_model, hybrid_model,dataset_details)
    #Measuring Robustness
    robustness_data = []
    if carlini_wagner_flag:
        attacks = ["clean","fgsm","pgd","cw"]
        perts = ["none","min","mean","mean"]
    else:
        attacks = ["clean","fgsm","pgd"]
        perts = ["none","min","mean"]    
    hashs_set, all_adversarial_examples, all_accuracies = [], [], [] 
    for classifier in classifiers:
        hash_value = generate_sha3_256_hash(classifier.model)
        hashs_set.append(hash_value)
        flag = check_redundancy(hash_value,dataset_name)
        if flag:
            adv_examples = load_from_memo(hash_value,dataset_name)
        else:
            adv_examples = generate_adversarial_attacks(classifier,x_test,epsilon,maximum_iterations)
            store_adv_examples(dataset_name,hash_value,adv_examples)
        all_adversarial_examples.append(adv_examples)
        if carlini_wagner_flag:
            perturbations = generate_pert(classifier,adv_examples[2],x_test,adv_examples[3])
        else:
            perturbations = generate_pert(classifier,adv_examples[2],x_test)
        for idx,(adv_example,perturbation) in enumerate(zip(adv_examples,perturbations)):
            metrics = evaluate_model(classifier,adv_example,y_test)
            all_accuracies.append(metrics[0])
            if not flag:
                robustness_data.append([hash_value,attacks[idx]]+list(metrics)+[perts[idx],perturbation])
    robustness_dataframe = pd.DataFrame(robustness_data,columns=["CNN_ID","attack_type","acc","precision","f1","auc-roc","pert_type","pert_size"])
    #Measuring Transferability
    transfer_data = []
    for i, attack in enumerate(attacks[1:]):
        n_attacks = len(attacks)
        metrics = evaluate_transferability(classifiers[1], all_adversarial_examples[0][i+1], y_test, all_accuracies[n_attacks+1+i])
        transfer_data.append([hashs_set[0], hashs_set[1], attack]+list(metrics))
        metrics2 = evaluate_transferability(classifiers[0], all_adversarial_examples[1][i+1], y_test, all_accuracies[1+i])
        transfer_data.append([hashs_set[1], hashs_set[0], attack]+list(metrics2))
    transfer_dataframe = pd.DataFrame(transfer_data,columns=["Donor_ID","Recipient_ID","attack_type","TSR","acc_drop"])
    return (robustness_dataframe,transfer_dataframe)
    