import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class activation_quant(Layer):
    def __init__(self, num_bits, max_value, **kwargs):
        super(activation_quant, self).__init__(**kwargs)
        self.num_bits = num_bits
        self.max_value = max_value
        
    def build(self, input_shape):
        self.relux = self.add_weight(name='relux',
                                     shape=[],
                                     initializer=keras.initializers.Constant(self.max_value))

    def call(self, x):
        act = tf.maximum(tf.minimum(x, self.relux), 0)
        act = tf.quantization.fake_quant_with_min_max_vars(
            act,
            min=tf.constant(0, dtype=tf.float32),
            max=self.relux,
            num_bits=self.num_bits,
            narrow_range=False)
        return act
    
    def compute_output_shape(self, input_shape):
        return input_shape

    
class conv2d_noise(Layer):
    def __init__(self, num_filter, kernel_size=3, activation=None, strides=1, padding='valid', noise_magnitude=0, **kwargs):
        super(conv2d_noise, self).__init__(**kwargs)
        self.num_filter = num_filter
        self.noise_magnitude = noise_magnitude
        self.kernel_size = (kernel_size, kernel_size)
        self.activation = activation
        self.strides = (strides, strides)
        self.padding = padding
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.kernel_size + (int(input_shape[3]), self.num_filter),
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=(self.num_filter,),
                                    initializer=keras.initializers.Zeros())
        
    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        weights = self.kernel
        bias = self.bias
        training = True
        if training:
            w_max = K.max(K.abs(weights))
            weights = weights + tf.random.normal(self.kernel.shape, mean=0, stddev=w_max * self.noise_magnitude)
            bias = bias + tf.random.normal(self.bias.shape, stddev=w_max * self.noise_magnitude)
        act = K.conv2d(x, weights, strides=self.strides, padding=self.padding)
        act = K.bias_add(act, bias)
        if self.activation == 'relu':
            act = K.relu(act)
        return act
    
    def compute_output_shape(self, input_shape):
        hei = conv_utils.conv_output_length(input_shape[1], self.kernel_size[0], self.padding, self.strides[0])
        wid = conv_utils.conv_output_length(input_shape[2], self.kernel_size[1], self.padding, self.strides[1])
        return (int(input_shape[0]), hei, wid, self.num_filter)

    
class dense_noise(Layer):
    def __init__(self, output_dim, activation=None, noise_magnitude=0, **kwargs):
        super(dense_noise, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.noise_magnitude = noise_magnitude
        self.activation = activation
        
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=[int(input_shape[1]), int(self.output_dim)],
                                      initializer='glorot_uniform')
        self.bias = self.add_weight(name='bias',
                                    shape=[int(self.output_dim)],
                                    initializer=keras.initializers.Zeros())
        
    def call(self, x, training=None):
        if training is None:
            training = K.learning_phase()
        weights = self.kernel
        bias = self.bias
        training = True
        if training:
            w_max = K.max(K.abs(weights))
            weights = weights + tf.random.normal(self.kernel.shape, mean=0, stddev=w_max * self.noise_magnitude)
            bias = bias + tf.random.normal(self.bias.shape, stddev=w_max * self.noise_magnitude)
        act = K.dot(x, weights) + bias
        if self.activation == 'relu':
            act = K.relu(act)
        elif self.activation == 'softmax':
            act = K.softmax(act)
        return act
    
    def compute_output_shape(self, input_shape):
        return (int(input_shape[0]), self.output_dim)