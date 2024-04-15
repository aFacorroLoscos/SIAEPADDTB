# Bibliotecas generales
import keras
import math
import numpy as np
import tensorflow as tf

# Graph plotting
from matplotlib import pyplot as plt

# Utilizado en las capas de Autoencoder
from keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from keras.layers import Layer, Input, InputSpec, BatchNormalization, Dropout, Dense
from keras import regularizers, activations, initializers,constraints, Sequential
from keras import backend as K

# Metricas
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Utilizado en funcion de entrenamiento
from sklearn.model_selection import train_test_split

def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))

class Autoencoder:

    """
        Pre: 
        Post: Creamos la clase autoencoder
    """
    def __init__(self,
                 input_shape,
                 layer_sizes,
                 latent_space,
                 drop_out_value,
                 activation_func):
        
        # Variables privadas
        self._input_shape = input_shape # Tamaño de entrada del modelo
        self._layer_sizes = layer_sizes
        self._num_layers = len(layer_sizes)
        self._latent_space = latent_space
        self._dropout_value = drop_out_value
        self._activation_func = activation_func

        # Capas del encoder
        self._encoder = None
        self._decoder = None

        # Modelos del Autoencoder
        self._model = None
        self._model_fine_tuning = None
        
        # Datos de entrenamiento del Autoencoder
        self._autoencoder_train = None
        self._autoencoder_fine_tuned_train = None

        # Entrada y salida para crear los modelos
        self._model_input = None
        self._encoder_output = None

        # Lista de capas del encoder para aplicarlas al decoder
        self._dense_list = []

        # Funcion build para crear el Autoencoder
        self._build()

    """
        Pre: ---
        Post: Inicializacion de los modelos Encoder, Decoder y Autoencoder
    """
    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    """
        Pre: ---
        Post: Creamos el modelo Encoder
    """
    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        encoder_layers = self._add_layers_encoder(encoder_input)
        encoder_output = self._add_encoder_output(encoder_layers)

        self._model_input = encoder_input

        self._encoder = Model(encoder_input, encoder_output, name = "Encoder_Layer")

    """
        Pre: ---
        Post: Creamos el modelo Decoder
    """
    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        decoder_layers = self._add_layers_decoder(decoder_input)
        decoder_output = self._add_decoder_output(decoder_layers)

        self._decoder = Model(decoder_input, decoder_output, name = "Decoder_Layer")
    """
        Pre: ---
        Post: Creamos el modelo Autoencoder utilizando el Encoder y Decoder
    """
    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self._decoder(self._encoder(model_input))

        self._model = Model(model_input, model_output, name = "Autoencoder_Model")

    """
        Pre: ---
        Post: Devolemos la capa input del Encoder
    """
    def _add_encoder_input(self):
        x =  keras.Input(shape = (self._input_shape,),
                         dtype = tf.float32, 
                         name = "Encoder_Input")
        return x
    
    """
        Pre: ---
        Post: Devolvemos la capa input del Decoder
    """
    def _add_decoder_input(self):
        x = keras.Input(shape=(self._latent_space,),
                        dtype = tf.float32, 
                        name = "Decoder_Input")
        return x
    
    """
        Pre: ---
        Post: Devolvemos la salida del Encoder
    """
    def _add_encoder_output(self,x):

        x = BatchNormalization()(x)

        dense = Dense(self._latent_space,
                      activation = self._activation_func, 
                      input_shape = (self._layer_sizes[-1],),
                      kernel_initializer = initializers.he_normal,
                      kernel_regularizer = keras.regularizers.L2(0.01),
                      use_bias = True)
        
        self._dense_list.append(dense)

        x = dense(x)

        x = Dropout(self._dropout_value)(x)

        return x
    
    """
        Pre: ---
        Post: Devolvemos la salida del Decoder
    """
    def _add_decoder_output(self,x):
        

        x = DenseTied(self._input_shape, 
                      activation = "linear",
                      use_bias = True,
                      tied_to = self._dense_list[0])(x)

        return x

    """
        Pre: ---
        Post: Anyadimos layers al Encoder
    """
    def _add_layers_encoder(self, encoder_input):
        x = encoder_input
        for layer_index in range(self._num_layers):
            input_dim = 0
            
            if layer_index == 0:
                input_dim = self._input_shape
            else:
                input_dim = self._layer_sizes[layer_index - 1]
            x = BatchNormalization()(x)

            x = self.__add_encoded_dense(self._layer_sizes[layer_index],
                                         input_dim,
                                         self._activation_func,
                                         x)
            
            x = Dropout(self._dropout_value)(x)

        return x
    
    """
        Pre: ---
        Post: Anyadimos layers al decoder
    """
    def _add_layers_decoder(self, decoder_input):
        x = decoder_input
        for layer_index in reversed(range(1, len(self._dense_list))):
            input_dim = self._layer_sizes[layer_index - 1]
            dense_tied = self._dense_list[layer_index]
            x = BatchNormalization()(x)

            x = self.__add_decoded__dense(input_dim,
                                          dense_tied,
                                          self._activation_func,
                                          x)
            
            # Capa BatchNormalization y Dropout
            x = Dropout(self._dropout_value)(x)

        return x

    """
        Pre: ---
        Post: Añadimos un dense al encoder
    """
    def __add_encoded_dense(self, node_size, input_dim, activation_func, x):
        dense = Dense(node_size, 
                      activation = activation_func,
                      input_shape = (input_dim,),
                      kernel_initializer = initializers.he_normal,
                      use_bias = True)

        self._dense_list.append(dense)
        x = dense(x)

        return x

    """
        Pre: ---
        Post: Añadimos dense al decoder
    """
    def __add_decoded__dense(self, input_dim, encoder_layer, activation_func, x):

        x = DenseTied(input_dim, 
                      activation = activation_func,
                      tied_to = encoder_layer,
                      use_bias = True)(x)
        
        return x

    def get_autoencoder_model(self):
        return self._model

    """
        Pre: numberOutputs es mayor que 0
        Post: Aplicamos el metodo Fine Tuning al autoencoder
    """
    def fine_tuning(self, numberOutputs):

        self._model_fine_tuning = Sequential()
        self._model_fine_tuning.add(keras.Input(shape = (self._input_shape,),
                                    dtype = tf.float32 ))
        
        for layer in self._encoder.layers:
            self._model_fine_tuning.add(layer)

        self._model_fine_tuning.add(Dense(numberOutputs, activation = 'softmax'))

    """
        Pre: trainble es una variable booleana
        Post: Las layer del modelo AE con fine tuning no cambian con el training
    """
    def set_trainable(self, trainable):
        for layer in self._model_fine_tuning.layers:
            layer.trainable = trainable

    """
        Pre: ---
        Post: Devuelve un resumen del autoencoder
    """
    def summary(self):
        self._encoder.summary()
        self._decoder.summary()
        self._model.summary()
    
    """
        Pre: ---
        Post: Devuelve un resumen del autoencoder tras aplicar Fine Tuning
    """
    def summary_fine_tuning(self):
        self._model_fine_tuning.summary()

    """
        Pre: learning_rate es mayor que 0
        Post: Compilamos el autoencoder
    """
    def compile(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        loss = keras.losses.MeanAbsoluteError()
        self._model.compile(optimizer = optimizer, loss = loss,
                           metrics = [tf.keras.metrics.MeanAbsoluteError()])

    """
        Pre: learning_rate es mayor que 0
        Post: Compilamos el autoencoder tras aplicar Fine Tuning
    """
    def compile_fine_tuning(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        loss = tf.keras.losses.CategoricalCrossentropy(reduction = "sum")
        self._model_fine_tuning.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy",
                             metrics=['sparse_categorical_accuracy'])

    """
        Pre: x_train es un conjunto de datos,
             batch_size y num_epochs es mayor que 0
        Post: Entrenamos el autoencoder
    """
    def train(self, x_train, batch_size, num_epochs, verbose_mode):
        
        train_X, valid_X, train_ground, valid_ground = train_test_split(x_train, x_train, 
                                                                        test_size = 0.33,
                                                                        shuffle = True, 
                                                                        random_state = 42)

        self._autoencoder_train = self._model.fit(train_X,
                       train_ground,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       verbose = verbose_mode,
                       shuffle = True,
                       validation_data = (valid_X, valid_ground))

        

    """
        Pre: x_train es un conjunto de datos,
             batch_size y num_epochs es mayor que 0
        Post: Entrenamos el autoencoder tras aplicar Fine Tuning
    """
    def train_fine_tuning(self, x_train, y_train, batch_size, num_epochs, verbose_mode):

        train_X, valid_X, train_label, valid_label = train_test_split(x_train, y_train, 
                                                                      test_size = 0.33,
                                                                      shuffle = True, 
                                                                      random_state = 42,
                                                                      stratify=y_train)
        
        self._autoencoder_fine_tuned_train = self._model_fine_tuning.fit(train_X, train_label,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       verbose = verbose_mode,
                       shuffle = True,
                       validation_data = (valid_X, valid_label))

    """
        Pre: x_Data es un conjunto de datos
        Post: Devolvemos la prediccion de los datos de x_Data
    """
    def predict(self, data):
        return self._model.predict(data)
    
    """
        Pre: x_Data es un conjunto de datos
        Post: Devolvemos la prediccion de los datos de x_Data
    """
    def predict_reduced(self, data):
        self._model = Model(inputs=self._model.input, outputs=self._model.get_layer("encoder").output)
        return self._model.predict(data)

    """
        Pre: x_Data es un conjunto de datos
        Post: Devolvemos la prediccion de los datos de x_Data
    """
    def predict_fine_tuning(self, data, verbose_mode):

        # Obtenemos la probabilidad de pertenecer a cada una de las clases
        y_pred = self._model_fine_tuning.predict(data, verbose = verbose_mode)
        result = np.zeros(y_pred.shape[0], dtype=int)

        # Devolvemos la clase con mayor probabilidad
        i = 0
        for probs in y_pred:
            probs = probs.tolist()  
            index = probs.index(max(probs))
            result[i] = index
            i += 1
        return result
    
    def predict_proba(self, xData, verboseMode):
        return self._model_fine_tuning.predict(xData, verbose = verboseMode)


    """
        Pre: x_Data es un conjunto de datos
        Post: Devolvemos un conjunto de datos pero con atributos reducidos
    """
    def return_reduce_attribute(self, x_Data):

        layers = self._model_fine_tuning.layers

        new_model = tf.keras.models.Sequential(layers[:-1])

        return new_model.predict(x_Data)
    
    def obtain_history(self, numModel, metricUsed, nameFile):
        model_trained = self._autoencoder_train
        metric_name = self._model.metrics_names[metricUsed]
        if(numModel == 1):
            model_trained = self._autoencoder_fine_tuned_train
            metric_name = self._model_fine_tuning.metrics_names[metricUsed]

        loss = model_trained.history[metric_name]
        val_loss = model_trained.history["val_" + metric_name]
        epochs = range(len(val_loss))
        plt.figure()
        plt.plot(epochs, loss, 'r', label='Training ' + metric_name)
        plt.plot(epochs, val_loss, 'b', label='Validation ' + metric_name)
        plt.title('Training and Validation ' + metric_name)
        plt.legend()
        plt.savefig(nameFile + '.png')
     
class DenseTranspose(keras.layers.Layer):
    """
        Pre: ---
        Post: Creamos la clase Dense Tranpose
    """
    def __init__(self,
                 dense,
                 activation = None, **kwargs):
        self.dense = dense
        self.activation = keras.activations.get(activation)
        self.kernel_regularizer = keras.regularizers.l2(0.001)
        super().__init__(**kwargs)
    
    """
        Pre: ---
        Post: Definimos el valor de los bias.
    """
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name = "bias",
                                      shape = [self.dense.input_shape[-1]],
                                      initializer = "zeros")
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b = True)
        return self.activation(z + self.biases)


class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] == self.units
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
