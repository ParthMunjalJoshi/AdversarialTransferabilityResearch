import entanglement_model as em
import evaluation_pipeline as ep

classical_model = em.entanglement_model_factory(5,0,1,["none"])
print(classical_model.summary())
hqcnn = em.entanglement_model_factory(5,4,3,["circularx"]*3)
print(hqcnn.summary())

