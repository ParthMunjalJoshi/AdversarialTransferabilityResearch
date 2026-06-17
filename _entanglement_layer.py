import tensorflow as tf
import _entanglement_circuit as ec

class EntanglementKerasLayer(tf.keras.layers.Layer):
    """Custom Keras layer integrating a quantum node (QNode) with TensorFlow.

    This layer wraps a PennyLane QNode into a trainable Keras layer, allowing hybrid
    quantum-classical model training using TensorFlow's backpropagation and optimizers.

    Attributes:
        qnode (callable): A PennyLane QNode expected to support TensorFlow interface.
        weight_shapes (dict): Dictionary specifying the shape of each trainable parameter.
        output_dim (int): Dimension of the output tensor.
        entanglement_info (Any): Encodes the entanglement structure to be used in the QNode.
        init_embed_rot (str): Initial rotation gate to apply to each qubit ('X', 'Y', or 'Z').
        depth (int): Number of repeated entangling layers in the quantum circuit.
    """

    def __init__(self, qnode, weight_shapes, output_dim, entg, embedded_rotation, depth, **kwargs):
        """Initializes the EntanglementKerasLayer.

        Args:
            qnode (callable): A PennyLane QNode using the 'tf' interface.
            weight_shapes (dict): A dictionary of trainable parameter names and their shapes.
            output_dim (int): The number of output features from the QNode.
            entg (Any): Information defining the entanglement structure of the circuit.
            embedded_rotation (str): Initial embedding gate.
            depth (int): Depth of the entanglement layers. 
            **kwargs: Additional keyword arguments for base Keras Layer.
        """
        super().__init__(**kwargs)
        self._qnode_name = qnode.__name__
        self._qnode_device_name = qnode.device.name
        self._qnode_interface = qnode.interface
        self.qnode = qnode
        self.weight_shapes = weight_shapes
        self.output_dim = output_dim
        self.entanglement_info = entg
        self.init_embed_rot = embedded_rotation
        self.depth = depth
        if self.qnode.interface != "tf":
            self.qnode.interface = "tf"

    def build(self, input_shape):
        """Creates the trainable weights for the quantum node.

        Args:
            input_shape (TensorShape): Shape of the input tensor.
        """
        self.qnode_weights = {}
        for weight_name, shape in self.weight_shapes.items():
            self.qnode_weights[weight_name] = self.add_weight(
                name=weight_name,
                shape=shape,
                initializer=tf.keras.initializers.RandomNormal(),
                trainable=True,
                dtype=tf.float32
            )
        super().build(input_shape)

    def call(self, inputs):
        """Executes the quantum circuit and returns its outputs.

        Args:
            inputs (tf.Tensor): Input tensor to be processed by the quantum circuit.

        Returns:
            tf.Tensor: Output tensor resulting from the QNode execution.
        """
        results = self.qnode(inputs, self.qnode_weights, self.entanglement_info, self.init_embed_rot)
        if isinstance(results, (list, tuple)):
            return tf.stack(results, axis=-1)
        return results

    def compute_output_shape(self, input_shape):
        """Computes the output shape of the layer.

        Args:
            input_shape (TensorShape): Shape of the input tensor.

        Returns:
            tuple: Output shape (batch_size, output_dim).
        """
        return (input_shape[0], self.output_dim)

    def get_config(self):
        """Returns the configuration of the layer for serialization.

        Returns:
            dict: Dictionary containing layer configuration.
        """
        config = super().get_config()
        config.update({
            'qnode': self._qnode_name,
            'weight_shapes': self.weight_shapes,
            'output_dim': self.output_dim,
            'entanglement_info': self.entanglement_info,
            'ckt_depth': self.depth,
            'angle_init': self.init_embed_rot
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Creates a layer instance from a configuration dictionary.

        Args:
            config (dict): Configuration dictionary as produced by `get_config()`.

        Returns:
            EntanglementKerasLayer: A new instance of this class.

        Raises:
            ValueError: If the specified QNode function is not recognized.
        """
        qnode_function_name = config.pop('qnode')
        if qnode_function_name == 'quantum_circuit':
            qnode_instance = ec.quantum_circuit
        else:
            raise ValueError(f"Unknown qnode function: {qnode_function_name}. "
                             "Please ensure the QNode is defined in the global scope "
                             "or accessible via a lookup mechanism.")
        return cls(qnode=qnode_instance, **config)
