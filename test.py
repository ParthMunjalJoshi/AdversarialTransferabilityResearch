import entanglement_model_factory as fac
import training_module as tm
import evaluation_pipeline as ep

dataset = "cifar10"

classical_model = fac.entanglement_model_factory((32,32,3),5,1,1,["classical"])
hybrid_model1 = fac.entanglement_model_factory((32,32,3),5,4,3,["circularx"]*3)
hybrid_model2 = fac.entanglement_model_factory((32,32,3),5,4,3,["none"]*3)

classical_model, _ = tm.train_model(classical_model,dataset)
hybrid_model1, _ = tm.train_model(hybrid_model1,dataset)
hybrid_model2, _ = tm.train_model(hybrid_model2,dataset)

rb1,tf1 = ep.eval_pipeline(dataset,classical_model,hybrid_model1,0.1,1,1)
rb2,tf2 = ep.eval_pipeline(dataset,classical_model,hybrid_model2,0.1,1,1)

print(rb1)
print(tf1)
print(rb2)
print(tf2)





