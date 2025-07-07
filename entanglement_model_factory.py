import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization
import _entanglement_layer as el
import _entanglement_circuit as ec
import _entanglement_strategies as es
import json

with open('lib/expt_config.json', 'r') as f:
    data = json.load(f)
n_qubits = data["quantum_ckt_parameters"]["n_qubits"]
depth=data["quantum_ckt_parameters"]["depth"]
n_conv_layers = data["convolutional_layers_parameters"]["n_layers"]
conv_layers = data["convolutional_layers_parameters"]["layers"]
assert len(conv_layers) == n_conv_layers

def retrieve_entg(n_qubits, depth, entanglement_details_list):
    """Parses entanglement strategy strings and constructs entanglement configuration.

    This function builds a list of entanglement strategy functions and their arguments 
    for each layer in the hybrid quantum-classical model.

    Args:
        n_qubits (int): Number of qubits (and size of quantum input layer).
        depth (int): Number of entanglement layers (circuit depth).
        entanglement_details_list (List[str]): List of entanglement strategies for each layer.
            Each entry must be one of:
            - 'classical': disables quantum processing
            - 'none': no entanglement
            - 'linearx', 'linearz': linear entanglement with CNOT or CZ
            - 'circularx', 'circularz': circular entanglement
            - 'fullx', 'fullz': fully connected entanglement
            - 'staggeredx', 'staggeredz': staggered pattern entanglement

    Returns:
        Tuple[bool, Tuple[List[Callable], List[Tuple]]]:
            - entg_flag (bool): Whether to use quantum entanglement (False if 'classical' is detected).
            - entg: A tuple of two lists:
                - List of entanglement functions (e.g., `es.linear_entanglement`)
                - List of corresponding argument tuples for each function
    """
    entg = [[], []]
    entg_flag = True
    for i in range(depth):
        if entanglement_details_list[i] == "classical":
            entg_flag = False
            break
        elif entanglement_details_list[i] == "none":
            entg[0].append(es.no_entanglement)
            entg[1].append(())
        elif entanglement_details_list[i] == "linearx":
            entg[0].append(es.linear_entanglement)
            entg[1].append((n_qubits, True))
        elif entanglement_details_list[i] == "linearz":
            entg[0].append(es.linear_entanglement)
            entg[1].append((n_qubits, False))
        elif entanglement_details_list[i] == "circularx":
            entg[0].append(es.circular_entanglement)
            entg[1].append((n_qubits, True))
        elif entanglement_details_list[i] == "circularz":
            entg[0].append(es.circular_entanglement)
            entg[1].append((n_qubits, False))
        elif entanglement_details_list[i] == "fullx":
            entg[0].append(es.full_entanglement)
            entg[1].append((n_qubits, True))
        elif entanglement_details_list[i] == "fullz":
            entg[0].append(es.full_entanglement)
            entg[1].append((n_qubits, False))
        elif entanglement_details_list[i] == "staggeredx":
            entg[0].append(es.staggered_entanglement)
            entg[1].append((n_qubits, True, i))
        elif entanglement_details_list[i] == "staggeredz":
            entg[0].append(es.staggered_entanglement)
            entg[1].append((n_qubits, False, i))
    return entg_flag, entg


def entanglement_model_factory(shape, n_qubits, num_parallel_filters, depth, entanglement_details_list,n_classes):
    """Constructs a hybrid quantum-classical convolutional neural network model.

    This model includes classical convolutional layers followed by one or more quantum layers
    using a configurable entanglement strategy. If classical-only mode is detected, the quantum
    layer is replaced by a dense classical substitute.

    Args:
        shape (Tuple[int, int, int]): Input image shape (e.g., (28, 28, 1) or (32, 32, 3)).
        n_qubits (int): Number of qubits used in the quantum layer.
        num_parallel_filters (int): Number of parallel quantum filters to apply.
        depth (int): Number of entanglement layers (quantum circuit depth).
        entanglement_details_list (List[str]): List of entanglement strategies for each depth layer.
            Same options as in `retrieve_entg`.

    Returns:
        keras.Model: A hybrid model with quantum or classical post-processing.
    """
    entg_flag, entanglement_details = retrieve_entg(n_qubits, depth, entanglement_details_list)
    weights = {"weights": (depth, 3, n_qubits)}
    
    x = Input(shape=shape)
    inputs = x
    for i in range(n_conv_layers):
        x = Conv2D(conv_layers[i]["filters"], tuple(conv_layers[i]["kernel_size"]), padding=conv_layers[i]["padding"], name=f"conv2d_{i}")(x)
        if conv_layers[i]["batch-norm"]:
            x = BatchNormalization(name=f"batch_norm_{i}")(x)
        x = Activation(conv_layers[i]["activation"], name=f"activation_{i}")(x)
        x = MaxPooling2D(tuple(conv_layers[i]["max_pool_kernel"]), name=f"max_pooling2d_{i}")(x)
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
                output_dim=n_qubits,
                entg=entanglement_details,
                embedded_rotation="X",
                depth=depth,
                name=f"quantum_filter_{i}"
            )(quantum_input)
            parallel_quantum_outputs.append(filter_layer)

        if num_parallel_filters > 1:
            concatenated_quantum_output = concatenate(parallel_quantum_outputs, axis=-1, name="concatenated_quantum_features")
        else:
            concatenated_quantum_output = parallel_quantum_outputs[0]

        outputs = Dense(n_classes, activation="softmax", name="output_classification")(concatenated_quantum_output)
    else:
        x = Dense(2 * n_qubits, name="middle_dense")(quantum_input)
        outputs = Dense(n_classes, activation="softmax", name="output_classification")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HQCNN")
    return model