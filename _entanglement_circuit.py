import pennylane as qml

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev,interface = "tf")
def quantum_circuit(inputs,weights,entg,embedded_rotation="Z",depth=3):
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
    


        


