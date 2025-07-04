import pennylane as qml
import tensorflow as tf
import json

with open('lib/expt_config.json', 'r') as f:
    data = json.load(f)
n_qubits = data["quantum_ckt_parameters"]["n_qubits"]
depth=data["quantum_ckt_parameters"]["depth"]

dev = qml.device("default.qubit", wires=n_qubits)
def validate_input(inputs,weights,entg,embedded_rotation,depth):
    """
    Validates inputs of quantum circuit.

    Args:
        inputs (array_like): values produced by previous model layer.
        weights (array_like): weights of given pqc.
        entg (list(list(fxns),list(args))): provides entanglement details of model.
        embedded_rotation (str): provides gates used for angle embedding.
        depth (int): provides depth of Ansatz.
     Returns:
        None
    Raises:
        ValueError: if depth is not a positive integer.
                    if embedded rotation is not X or Y or Z.
                    if weights are not in apt shape.
                    if length of entanglement fxns or args is not depth
        TypeError:  if inputs is not array-like with length = n_qubits
                    if entg is not a tuple of fxns,args
                    if entg fxns are not callable            
        KeyError:   if weights doesn't have the proper key 'weights'.
    """
    # Validate depth
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError("`depth` must be a positive integer. Check config file")
    # Validate embedded_rotation
    embedded_rotation = str(embedded_rotation).upper()
    if embedded_rotation not in {"X", "Y", "Z"}:
        raise ValueError("`embedded_rotation` must be 'X', 'Y', or 'Z'.")
    # Validate weights
    if not isinstance(weights, dict) or "weights" not in weights:
        raise KeyError("`weights` must be a dictionary containing a 'weights' key.")
    weights_array = weights["weights"]
    weights_array = tf.convert_to_tensor(weights_array)  # Ensures TensorFlow compatibility
    if weights_array.shape != (depth, 3, n_qubits):
        raise ValueError(f"`weights['weights']` must have shape ({depth}, 3, {n_qubits}). Got {weights_array.shape}.")
    # Validate entanglement structure
    if not isinstance(entg, (list, tuple)) or len(entg) != 2:
        raise TypeError("`entg` must be a tuple or list of two elements: (functions, args).")
    entg_fxns, entg_args = entg
    if not (len(entg_fxns) == len(entg_args) == depth):
        raise ValueError(f"Length of entanglement functions and arguments must match depth ({depth}).")
    for idx, fxn in enumerate(entg_fxns):
        if not callable(fxn):
            raise TypeError(f"entg_fxns[{idx}] is not callable. Got type: {type(fxn)}")
    

@qml.qnode(dev,interface = "tf")
def quantum_circuit(inputs,weights,entg,embedded_rotation):
    """
    Defines a Parametrized Quantum Circuit given information about entanglement layers.

    Args:
        inputs (array_like): values produced by previous model layer.
        weights (array_like): weights of given pqc.
        entg (list(list(fxns),list(args))): provides entanglement details of model.
        embedded_rotation (str): provides gates used for angle embedding.
     Returns:
        measured (array_like) : measured PauliZ values of each of the qubits.
    Raises:
        ValueError: if depth is not a positive integer.
                    if embedded rotation is not X or Y or Z.
                    if weights are not in apt shape.
                    if length of entanglement fxns or args is not depth
        TypeError:  if inputs is not array-like with length = n_qubits
                    if entg is not a tuple of fxns,args
                    if entg fxns are not callable            
        KeyError:   if weights doesn't have the proper key 'weights'.
    """
    validate_input(inputs,weights,entg,embedded_rotation,depth)
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits),rotation = embedded_rotation)
    entg_fxns,entg_args = entg
    for i in range(depth):
        w = weights["weights"][i]
        rz,ry,rz2 = w[0],w[1],w[2]
        for j in range(n_qubits):
            qml.RZ(rz[j], wires=j)
            qml.RY(ry[j], wires=j)
            qml.RZ(rz2[j], wires=j)
        entg_fxns[i](*entg_args[i])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]       
    


        


