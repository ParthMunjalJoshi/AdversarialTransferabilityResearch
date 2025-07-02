import pennylane as qml

#Warning: replacing this number to actual amount is essential for proper functioning
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev,interface = "tf")
def quantum_circuit(inputs,weights,entg,embedded_rotation,depth=3):
    """
    Defines a circuit given information about entanglement layers.

    Args:
        inputs (array_like): values produced by previous model layer.
        weights (array_like): weights of given pqc.
        entg (list(list(fxns),list(args))): provides entanglement details of model.
        embedded_rotation (str): provides gates used for angle embedding.
        depth (int): provides depth of Ansatz.
     Returns:
        measured (array_like) : measured PauliZ values of each of the qubits.
    Raises:
        ValueError: if embedded_rotation is not X,Y or Z.

    """
    if(str(embedded_rotation).upper() not in ["X","Y","Z"]):
        raise ValueError("embedded_rotation must be X or Y or Z only")

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
    


        


