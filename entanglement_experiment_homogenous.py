import entanglement_model_factory as fac
import entanglement_training_module as tm
import evaluation_pipeline as ep
import os
import pickle
import pandas as pd
import json
import hashlib
import clear_temp_folder as ctf
from tensorflow import keras

with open('lib/expt_config.json', 'r') as f:
    data = json.load(f)

dependancies = ["quantum_ckt_parameters","convolutional_layers_parameters","training_parameters"]
dict_str = b''
for d in dependancies:
    model_associated_data = data[d]
    dict_str += json.dumps(model_associated_data, sort_keys=True).encode('utf-8')
config_hash = hashlib.sha3_256(dict_str).hexdigest()

n_qubits = data["quantum_ckt_parameters"]["n_qubits"]
depth=data["quantum_ckt_parameters"]["depth"]
num_parallel_filters = data["quantum_ckt_parameters"]["num_parallel_filters"]
datasets = data["dataset_details"]["datasets"]
shapes = data["dataset_details"]["shapes"]
num_classes = data["dataset_details"]["output_classes"]
assert len(datasets) == len(shapes) == len(num_classes)
epsilon = data["adversarial_parameters"]["epsilon"]
max_iter = data["adversarial_parameters"]["max_iter"]
subset_size = data["adversarial_parameters"]["subset_size"]
clr_flag = data["clear_temp_file_after_expt"]

def save_history(history,path):
    """Saves training history via pickling.

    Args:
        history (keras.callbacks.History): Training history object.
        path (str): path to save history as a pickle file.
    """
    with open(path, 'wb') as histfile:
        pickle.dump(history, histfile) 

def append_to_db(path_csv,df):
    """Appends a DataFrame to a CSV file. If the file doesn't exist, creates it.

    Args:
        path_csv (str): Path to the CSV file.
        df (pd.DataFrame): DataFrame to append.
    """
    if os.path.exists(path_csv):
        with open(path_csv, 'a', newline='') as f:
            df.to_csv(f, header=False, index=False)
    else:
        df.to_csv(path_csv,header=True,index = False)

def main():
    """Main orchestration function for automated model evaluation.

    For each dataset (MNIST, FMNIST, CIFAR-10), this function:
    1. Trains a classical CNN model.
    2. Iterates over multiple entanglement strategies.
    3. Builds, trains, and evaluates corresponding HQCNN models.
    4. Measures robustness and transferability using adversarial examples.
    5. Appends results to persistent CSV files:
        - `lib/robustness.csv`: Adversarial robustness metrics.
        - `lib/transferability.csv`: Attack transfer metrics.
        - `lib/index.csv`: Lookup for model names and their corresponding hashes.
    """
    import tensorflow as tf
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    
    column_names = ["model","hash"]
    os.makedirs("lib/models", exist_ok=True)
    os.makedirs("lib/hist", exist_ok=True)
    for idx,dataset in enumerate(datasets):
        print(f"Running Classical Model Code on {dataset}. ")
        classical_path = "lib/models/classical_"+dataset+str(config_hash)+".keras" 
        classical_history_path = "lib/hist/classical_"+dataset+str(config_hash)+".pkl"
        if os.path.exists(classical_path):
            try:
                classical_model = keras.models.load_model(classical_path)  
            except Exception as e:
                print(f"Unable to load classical model due to : {str(e)}")  
        else:
            print(f"Training classical model on {dataset}.")
            classical_model = fac.entanglement_model_factory(shapes[idx],n_qubits,num_parallel_filters,depth,["classical"],num_classes[idx])
            classical_model, classical_history = tm.train_model(classical_model,dataset)
            try:
                classical_model.save(classical_path)
                save_history(classical_history,classical_history_path)
            except Exception as e:
                print(f"Unable to save classical model/history due to : {str(e)}")  
        entanglement_strategies = [ i*depth for  i in [["none"],["linearx"],["linearz"],["circularx"],["circularz"],["fullx"],["fullz"],["staggeredx"],["staggeredz"]]]
        for entanglement_strategy in entanglement_strategies:
            index = []
            #Generating path compatible strategy name (avoid '[]' and ',' .etc)
            strategy_name = "_".join(entanglement_strategy)
            print(f"Running Hybrid Model Code on {dataset} with {strategy_name}.")
            hybrid_path = f"lib/models/hybrid_{dataset}_{strategy_name}_{str(config_hash)}.weights.h5"
            hybrid_history_path = f"lib/hist/hybrid_{dataset}_{strategy_name}_{str(config_hash)}.pkl"
            hybrid_model = fac.entanglement_model_factory(shapes[idx],n_qubits,num_parallel_filters,depth,entanglement_strategy,num_classes[idx])
            if os.path.exists(hybrid_path):
                try:
                    hybrid_model.load_weights(hybrid_path)
                except Exception as e:
                    print(f"Unable to load hybrid model due to : {str(e)}") 
            else:
                print(f"Training hybrid model with {strategy_name} on {dataset}. ")
                hybrid_model, hybrid_history = tm.train_model(hybrid_model,dataset)
                try:
                    hybrid_model.save_weights(hybrid_path)
                    save_history(hybrid_history,hybrid_history_path)
                except Exception as e:
                    print(f"Unable to save hybrid model/history due to : {str(e)}")  
            rb,tf = ep.eval_pipeline(dataset,classical_model,hybrid_model,epsilon,max_iter,subset_size)
            try:
                append_to_db("lib/results/robustness.csv",rb)
                append_to_db("lib/results/transferability.csv",tf)
            except Exception as e:
                print(f"Unable to save results to csv files due to : {str(e)}")
            if "CNN_ID" in rb.columns:
                ids = rb["CNN_ID"].unique()
                if len(ids) >= 2:
                    index.append(["CNN_" + dataset, ids[0]])
                    index.append(["HQCNN_" + dataset + "_" + str(entanglement_strategy), ids[1]])
                elif len(ids) == 1:
                    index.append(["HQCNN_" + dataset + "_" + str(entanglement_strategy), ids[0]])
            index_df = pd.DataFrame(index,columns=column_names)
            append_to_db("lib/results/index.csv",index_df)

if __name__ == "__main__":
    main()
    if clr_flag:
        #Cleanup generated adv-examples after end of expt
        ctf.temp_clr()
