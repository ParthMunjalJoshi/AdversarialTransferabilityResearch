import pennylane as qml

def get_fxn(cnot_flag):
    """Returns the appropriate two-qubit gate based on the control flag.

    Args:
        cnot_flag (bool): If True, returns `qml.CNOT`. Otherwise, returns `qml.CZ`.

    Returns:
        function: A PennyLane quantum gate function (`qml.CNOT` or `qml.CZ`).
    """
    return qml.CNOT if cnot_flag else qml.CZ

def no_entanglement():
    """Placeholder function representing absence of entanglement.

    This function performs no operation and is used as a stand-in when no entanglement is required.
    """
    pass

def linear_entanglement(n_qubits, cnot_flag):
    """Applies linear entanglement across adjacent qubits in a quantum circuit.

    Qubits are entangled in a chain: (0,1), (1,2), ..., (n-2, n-1).

    Args:
        n_qubits (int): Number of qubits in the circuit.
        cnot_flag (bool): If True, uses CNOT gates; otherwise, uses CZ gates.
    """
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits - 1):
        fxn(wires=[i, i + 1])

def circular_entanglement(n_qubits, cnot_flag):
    """Applies circular entanglement, connecting each qubit to its next neighbor with wrap-around.

    Qubits are entangled in a ring: (0,1), (1,2), ..., (n-2, n-1), (n-1, 0).

    Args:
        n_qubits (int): Number of qubits in the circuit.
        cnot_flag (bool): If True, uses CNOT gates; otherwise, uses CZ gates.
    """
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits):
        fxn(wires=[i, (i + 1) % n_qubits])

def full_entanglement(n_qubits, cnot_flag):
    """Applies full entanglement between all unique pairs of qubits.

    Every qubit is entangled with every other qubit once.

    Args:
        n_qubits (int): Number of qubits in the circuit.
        cnot_flag (bool): If True, uses CNOT gates; otherwise, uses CZ gates.
    """
    fxn = get_fxn(cnot_flag)
    for i in range(n_qubits - 1):
        for j in range(i + 1, n_qubits):
            fxn(wires=[i, j])

def staggered_entanglement(n_qubits, cnot_flag, layers_done):
    """Applies staggered (alternating) entanglement pattern between adjacent qubits.

    Entanglement alternates between even and odd indexed pairs based on the current layer:
    - Even layers: (0,1), (2,3), ...
    - Odd layers: (1,2), (3,4), ...

    Args:
        n_qubits (int): Number of qubits in the circuit.
        cnot_flag (bool): If True, uses CNOT gates; otherwise, uses CZ gates.
        layers_done (int): The number of layers already applied; determines offset.
    """
    fxn = get_fxn(cnot_flag)
    if layers_done % 2 == 0:
        for i in range(0, n_qubits - 1, 2):
            fxn(wires=[i, i + 1])
    else:
        for i in range(1, n_qubits - 1, 2):
            fxn(wires=[i, i + 1])
