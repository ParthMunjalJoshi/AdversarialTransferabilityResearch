import entanglement_model_factory as fac
import training_module as tm
import evaluation_pipeline as ep
import os
import pandas as pd

def append_to_db(path_csv,df):
    if not os.path.exists(path_csv):
        df.to_csv(path_csv, index=False)
    else:
        df.to_csv(path_csv, mode='a', header=False, index=False)

def main():
    datasets = ["mnist","fmnist","cifar10"]
    index = []
    shapes = {
        "mnist":(28,28,1),
        "fmnist":(28,28,1),
        "cifar10":(32,32,3)
    }
    column_names = ["model","hash"]
    for dataset in datasets:
        classical_model = fac.entanglement_model_factory(shapes[dataset],5,1,1,["classical"])
        classical_model, _ = tm.train_model(classical_model,dataset)
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
            rb,tf = ep.eval_pipeline(dataset,classical_model,hybrid_model,0.01,40,1000)
            append_to_db("lib/robustness.csv",rb)
            append_to_db("lib/transferability.csv",tf)
            index.append([["CNN_"+dataset] , rb["CNN_ID"].unique()[0]])
            index.append([["HQCNN_"+dataset+"_"+str(entanglement_strategy)] , rb["CNN_ID"].unique()[1]])
            index_df = pd.DataFrame(index,column=column_names)
            append_to_db("lib/index.csv",index_df)

if __name__ == "__main__":
    main()
