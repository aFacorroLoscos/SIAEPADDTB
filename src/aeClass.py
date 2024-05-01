# Bibliotecas generales
import tensorflow.keras
import math
import numpy as np
import tensorflow as tf

# Graph plotting
from matplotlib import pyplot as plt

# Utilizado en las capas de Autoencoder
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, Input, InputSpec, BatchNormalization, Dropout, Dense
from tensorflow.keras import regularizers, activations, initializers,constraints, Sequential
from tensorflow.keras import backend as K

# Metricas
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# Utilizado en funcion de entrenamiento
from sklearn.model_selection import train_test_split


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

        # Encoder y Decoder
        self._encoder = None
        self._decoder = None

        # Entrada y salida para crear los modelos
        self._model_input = None
        self._encoder_output = None

        # Modelos del Autoencoder
        self._model = None
        self._model_fine_tuning = None
        
        # Datos de entrenamiento del Autoencoder
        self._autoencoder_train = None
        self._autoencoder_fine_tuned_train = None


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
        x =  Input(shape = (self._input_shape,),
                         dtype = tf.float32, 
                         name = "Encoder_Input")
        return x
    
    """
        Pre: ---
        Post: Devolvemos la capa input del Decoder
    """
    def _add_decoder_input(self):
        x = Input(shape=(self._latent_space,),
                        dtype = tf.float32, 
                        name = "Decoder_Input")
        return x
    
    """
        Pre: ---
        Post: Devolvemos la salida del Encoder
    """
    def _add_encoder_output(self,x):
        
        input_encoder = (self._layer_sizes[-1],) if len(self._layer_sizes) != 0 else (self._input_shape,)

        x = Dense(self._latent_space,
                      activation = self._activation_func, 
                      input_shape = input_encoder,
                      kernel_initializer = initializers.he_normal,
                      kernel_regularizer = tensorflow.keras.regularizers.L2(0.01),
                      use_bias = True)(x)

        x = BatchNormalization()(x)

        x = Dropout(self._dropout_value)(x)

        return x
    
    """
        Pre: ---
        Post: Devolvemos la salida del Decoder
    """
    def _add_decoder_output(self,x):
        

        x = DenseTied(self._input_shape, 
                      activation = "sigmoid",
                      use_bias = True,
                      tied_to = self._encoder.layers[1])(x)

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
            
            x = self.__add_encoded_dense(self._layer_sizes[layer_index],
                                         input_dim,
                                         self._activation_func,
                                         x)

            x = BatchNormalization()(x)

            x = Dropout(self._dropout_value)(x)

        return x
    
    """
        Pre: ---
        Post: Anyadimos layers al decoder
    """
    def _add_layers_decoder(self, decoder_input):
        x = decoder_input

        for layer_index in reversed(range(1, self._num_layers + 1)):
            input_dim = self._layer_sizes[layer_index - 1]
            x = self.__add_decoded__dense(input_dim,
                                          self._encoder.layers[1 + layer_index * 3],
                                          self._activation_func,
                                          x)

            # Capa BatchNormalization y Dropout

            x = BatchNormalization()(x)

            x = Dropout(self._dropout_value)(x)

        return x

    """
        Pre: ---
        Post: Añadimos un dense al encoder
    """
    def __add_encoded_dense(self, node_size, input_dim, activation_func, x):

        x = Dense(node_size, 
                      activation = activation_func,
                      input_shape = (input_dim,),
                      kernel_initializer = initializers.he_normal,
                      use_bias = True)(x)

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
        self._model_fine_tuning.add(Input(shape = (self._input_shape,),
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
        loss = tensorflow.keras.losses.MeanAbsoluteError()
        self._model.compile(optimizer = optimizer, loss = loss,
                           metrics = [tensorflow.keras.metrics.MeanAbsoluteError()])

    """
        Pre: learning_rate es mayor que 0
        Post: Compilamos el autoencoder tras aplicar Fine Tuning
    """
    def compile_fine_tuning(self, learning_rate = 0.0001):
        optimizer = Adam(learning_rate = learning_rate)
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(reduction = "sum")
        self._model_fine_tuning.compile(optimizer = optimizer, loss = loss,
                             metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy()])

    """
        Pre: x_train es un conjunto de datos,
             batch_size y num_epochs es mayor que 0
        Post: Entrenamos el autoencoder
    """
    def train(self, x_train, batch_size, num_epochs, verbose_mode):
        
        train_X, valid_X, train_ground, valid_ground = train_test_split(x_train, x_train, 
                                                                        test_size = 0.3,
                                                                        shuffle = True, 
                                                                        random_state = 42)

        callback =  tensorflow.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10, min_delta = 0.01)

        
        
        self._autoencoder_train = self._model.fit(train_X,
                       train_ground,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       verbose = verbose_mode,
                       shuffle = True,
                       callbacks = [callback],
                       validation_data = (valid_X, valid_ground))

        

    """
        Pre: x_train es un conjunto de datos,
             batch_size y num_epochs es mayor que 0
        Post: Entrenamos el autoencoder tras aplicar Fine Tuning
    """
    def train_fine_tuning(self, x_train, y_train, batch_size, num_epochs, verbose_mode):

        train_X, valid_X, train_label, valid_label = train_test_split(x_train, y_train, 
                                                                      test_size = 0.3,
                                                                      shuffle = True, 
                                                                      random_state = 42,
                                                                      stratify=y_train)

        callback =  tensorflow.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = "min", verbose = 1, patience = 10)
        
        self._autoencoder_fine_tuned_train = self._model_fine_tuning.fit(train_X, train_label,
                       batch_size = batch_size,
                       epochs = num_epochs,
                       verbose = verbose_mode,
                       shuffle = True,
                       callbacks = [callback],
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

        return self._encoder.predict(data)

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

        new_model = tensorflow.keras.models.Sequential(layers[:-1])

        return new_model.predict(x_Data)
    
    def get_encoder_weights(self, layer_index):
        return self._encoder.layers[1 +  3 * layer_index].get_weights()

    def obtain_history(self, autoencoder_type):
        model_trained = self._autoencoder_train
        metrics_name = self._model.metrics_names

        if(autoencoder_type == "fine_tuning"):
            model_trained = self._autoencoder_fine_tuned_train
            metrics_name = self._model_fine_tuning.metrics_names

        for metric_name in metrics_name:
            loss = model_trained.history[metric_name]
            val_loss = model_trained.history["val_" + metric_name]
            epochs = range(len(val_loss))
            plt.figure()
            plt.plot(epochs, loss, 'r', label='Training ' + metric_name)
            plt.plot(epochs, val_loss, 'b', label='Validation ' + metric_name)
            plt.legend()
            plt.savefig("../results/train_metrics/" + metric_name + "_" + autoencoder_type + '.png')
     
class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 bias_regularizer=None,
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

        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
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
