import tensorflow as tf
import entanglement_circuit as ec

class QuantumKerasLayer(tf.keras.layers.Layer):
    def __init__(self, qnode, weight_shapes, output_dim,entg,embedded_rotation="Z",depth=3,**kwargs):
        super().__init__(**kwargs)
        self._qnode_name = qnode.__name__
        self._qnode_device_name = qnode.device.name
        self._qnode_interface = qnode.interface
        self.qnode = qnode
        self.weight_shapes = weight_shapes
        self.output_dim = output_dim
        self.entanglement_info = entg
        self.init_embed = embedded_rotation
        self.depth = depth
        if self.qnode.interface != "tf":
            self.qnode.interface = "tf"

    def build(self, input_shape):
        self.qnode_weights = {}
        for weight_name,shape in self.weight_shapes.items():
            self.qnode_weights[weight_name] = self.add_weight(
                name=weight_name,
                shape=shape,
                initializer=tf.keras.initializers.RandomNormal(),
                trainable=True,
                dtype=tf.float32
            )
        super().build(input_shape)

    def call(self, inputs):
        results = self.qnode(inputs, self.qnode_weights,self.entanglement_info,self.init_embed,self.depth)
        if isinstance(results, (list, tuple)):
            return tf.stack(results, axis=-1)
        return results


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            'qnode': self._qnode_name,
            'weight_shapes': self.weight_shapes,
            'output_dim': self.output_dim,
            'entanglement_info': self.entanglement_info,
            'ckt_depth':self.depth,
            'angle_init':self.init_embed
        })
        return config

    @classmethod
    def from_config(cls, config):
        qnode_function_name = config.pop('qnode')
        if qnode_function_name == 'quantum_circuit':
            qnode_instance = ec.quantum_circuit
        else:
            raise ValueError(f"Unknown qnode function: {qnode_function_name}. "
                             "Please ensure the QNode is defined in the global scope "
                             "or accessible via a lookup mechanism.")
        return cls(qnode=qnode_instance, **config)