import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization
import entanglement_layer as el
import entanglement_circuit as ec
import entanglement_strategies as es

def retrieve_entg(n_qubits,depth,entanglement_details_list):
    entg = [[],[]]
    entg_flag = True
    for i in range(depth):
        if(entanglement_details_list[i] == "none"):
            entg_flag = False
        elif(entanglement_details_list[i] == "linearx"):
            entg[0].append(es.linear_entanglement)
            entg[1].append((n_qubits,True))
        elif(entanglement_details_list[i] == "linearz"):
            entg[0].append(es.linear_entanglement)
            entg[1].append((n_qubits,False))
        elif(entanglement_details_list[i] == "circularx"):
            entg[0].append(es.circular_entanglement)
            entg[1].append((n_qubits,True))
        elif(entanglement_details_list[i] == "circularz"):
            entg[0].append(es.circular_entanglement)
            entg[1].append((n_qubits,False))
        elif(entanglement_details_list[i] == "fullx"):
            entg[0].append(es.full_entanglement)
            entg[1].append((n_qubits,True))
        elif(entanglement_details_list[i] == "fullz"):
            entg[0].append(es.full_entanglement)
            entg[1].append((n_qubits,False))
        elif(entanglement_details_list[i] == "staggeredx"):
            entg[0].append(es.staggered_entanglement)
            entg[1].append((n_qubits,True,i))
        elif(entanglement_details_list[i] == "staggeredz"):
            entg[0].append(es.staggered_entanglement)
            entg[1].append((n_qubits,False,i))
    return entg_flag,entg

def entanglement_model_factory(n_qubits,num_parallel_filters,depth,entanglement_details_list):
    entg_flag,entanglement_details = retrieve_entg(n_qubits,depth,entanglement_details_list)
    output_dim_per_filter = n_qubits
    weights = {"weights":(depth,3,n_qubits)}
    inputs = Input(shape=(28, 28, 1), name="input_image")

    x = Conv2D(16, (5, 5), padding="same", name="conv2d_1")(inputs)
    x = BatchNormalization(name="batch_norm_1")(x) 
    x = Activation('relu', name="activation_1")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_1")(x)

    x = Conv2D(32, (5, 5), padding="same", name="conv2d_2")(x)
    x = BatchNormalization(name="batch_norm_2")(x) 
    x = Activation('relu', name="activation_2")(x)
    x = MaxPooling2D((2, 2), name="max_pooling2d_2")(x)

    x = Flatten(name="flatten")(x)
    quantum_input = Dense(n_qubits, name="dense_to_quantum_input")(x)
    quantum_input = BatchNormalization(name="batch_norm_dense_to_quantum_input")(quantum_input) 
    quantum_input = Activation("relu", name="activation_dense_to_quantum_input")(quantum_input)

    if entg_flag:
        parallel_quantum_outputs = []
        for i in range(num_parallel_filters):
            filter_layer = el.EntanglementKerasLayer(
                qnode=ec.quantum_circuit,
                weight_shapes=weights,
                output_dim=output_dim_per_filter,
                entg = entanglement_details,
                depth = depth,
                name=f"quantum_filter_{i}"
            )(quantum_input)
            parallel_quantum_outputs.append(filter_layer)

        if num_parallel_filters > 1:
            concatenated_quantum_output = concatenate(parallel_quantum_outputs, axis=-1, name="concatenated_quantum_features")
        else:
            concatenated_quantum_output = parallel_quantum_outputs[0]
        outputs = Dense(10, activation="softmax", name="output_classification")(concatenated_quantum_output)
    else:
        x = Dense(2*n_qubits, name="middle_dense")(quantum_input)
        outputs = Dense(10, activation="softmax", name="output_classification")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HQCNN")
    return model



