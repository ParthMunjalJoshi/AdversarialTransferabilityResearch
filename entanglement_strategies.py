import pennylane as qml

def get_fxn(cnot_flag):
    return  qml.CNOT if cnot_flag else qml.CZ

def linear_entanglement(n_qubits,cnot_flag):
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits-1):
        fxn(wires=[i,i+1])

def circular_entanglement(n_qubits,cnot_flag):
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits):
        fxn(wires=[i,(i+1)%n_qubits])

def full_entanglement(n_qubits,cnot_flag):
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits-1):
        for j in range(i+1,n_qubits):
            fxn(wires=[i,j])

def staggered_entanglement(n_qubits,cnot_flag,layers_done):
    fxn = get_fxn(cnot_flag)
    if(layers_done%2 == 0):
        for i in range(0,n_qubits-1,2):
            fxn(wires=[i,i+1])
    else:
        for i in range(1,n_qubits-1,2):
            fxn(wires=[i,i+1])
    
    

