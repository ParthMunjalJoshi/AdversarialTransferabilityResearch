import pennylane as qml

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev,interface = "tf")
def quantum_circuit(inputs,weights,entg,embedded_rotation="Z",depth=3):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits),rotation = embedded_rotation)
    for i in range(depth):
        rz,ry,rz2 = weights[i]
        for j in range(n_qubits):
            qml.RZ(rz[j], wires=j)
            qml.RY(ry[j], wires=j)
            qml.RZ(rz2[j], wires=j)
        entanglement_layers,entanglement_args = entg
        entanglement_layers[i](entanglement_args)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]       
    
    

        


