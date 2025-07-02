import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, concatenate, BatchNormalization
import _entanglement_layer as el
import _entanglement_circuit as ec
import _entanglement_strategies as es

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


def entanglement_model_factory(shape, n_qubits, num_parallel_filters, depth, entanglement_details_list):
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
        keras.Model: A compiled hybrid model with quantum or classical post-processing.
    """
    entg_flag, entanglement_details = retrieve_entg(n_qubits, depth, entanglement_details_list)
    output_dim_per_filter = n_qubits
    weights = {"weights": (depth, 3, n_qubits)}
    
    inputs = Input(shape=shape, name="input_image")

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
                entg=entanglement_details,
                depth=depth,
                name=f"quantum_filter_{i}"
            )(quantum_input)
            parallel_quantum_outputs.append(filter_layer)

        if num_parallel_filters > 1:
            concatenated_quantum_output = concatenate(parallel_quantum_outputs, axis=-1, name="concatenated_quantum_features")
        else:
            concatenated_quantum_output = parallel_quantum_outputs[0]

        outputs = Dense(10, activation="softmax", name="output_classification")(concatenated_quantum_output)
    else:
        x = Dense(2 * n_qubits, name="middle_dense")(quantum_input)
        outputs = Dense(10, activation="softmax", name="output_classification")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HQCNN")
    return model
