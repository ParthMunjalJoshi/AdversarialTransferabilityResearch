import entanglement_model_factory as fac
import entanglement_training_module as tm
import evaluation_pipeline as ep

dataset = "mnist"

classical_model = fac.entanglement_model_factory((28,28,1),5,1,1,["classical"],10)
hybrid_model1 = fac.entanglement_model_factory((28,28,1),5,4,3,["circularx"]*3,10)

classical_model, _ = tm.train_model(classical_model,dataset)
hybrid_model1, _ = tm.train_model(hybrid_model1,dataset)

rb1,tf1 = ep.eval_pipeline(dataset,classical_model,hybrid_model1,0.1,1,1)

print(rb1)
print(tf1)