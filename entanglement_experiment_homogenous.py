import entanglement_model_factory as fac
import training_module as tm
import evaluation_pipeline as ep
import os
import pandas as pd

def append_to_db(path_csv,df):
    """Appends a DataFrame to a CSV file. If the file doesn't exist, creates it.

    Args:
        path_csv (str): Path to the CSV file.
        df (pd.DataFrame): DataFrame to append.
    """
    if not os.path.exists(path_csv):
        df.to_csv(path_csv, index=False)
    else:
        df.to_csv(path_csv, mode='a', header=False, index=False)

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
    datasets = ["mnist","fmnist","cifar10"]
    index = []
    shapes = {
        "mnist":(28,28,1),
        "fmnist":(28,28,1),
        "cifar10":(32,32,3)
    }
    column_names = ["model","hash"]
    os.makedirs(os.path.dirname("models"), exist_ok=True)
    for dataset in datasets:
        classical_model = fac.entanglement_model_factory(shapes[dataset],5,1,1,["classical"])
        classical_model, _ = tm.train_model(classical_model,dataset)
        classical_model.save("models/classical_"+dataset+".keras")
        entanglement_strategies = [
            ["none"]*3,
            ["linearx"]*3,
            ["linearz"]*3,
            ["circularx"]*3,
            ["circularz"]*3,
            ["fullx"]*3,
            ["fullz"]*3,
            ["staggeredx"]*3,
            ["staggeredz"]*3
        ]
        for entanglement_strategy in entanglement_strategies:
            hybrid_model = fac.entanglement_model_factory(shapes[dataset],5,4,3,entanglement_strategy)
            hybrid_model, _ = tm.train_model(hybrid_model,dataset)
            hybrid_model.save_weights("models/hybrid_"+dataset+entanglement_strategy+".weights.h5")
            rb,tf = ep.eval_pipeline(dataset,classical_model,hybrid_model,0.01,40,1000)
            append_to_db("lib/robustness.csv",rb)
            append_to_db("lib/transferability.csv",tf)
            index.append([["CNN_"+dataset] , rb["CNN_ID"].unique()[0]])
            index.append([["HQCNN_"+dataset+"_"+str(entanglement_strategy)] , rb["CNN_ID"].unique()[1]])
            index_df = pd.DataFrame(index,columns=column_names)
            append_to_db("lib/index.csv",index_df)

if __name__ == "__main__":
    main()
