# Bibliotecas generales
import tensorflow.keras
import numpy as np
import tensorflow as tf

# Graph plotting
from matplotlib import pyplot as plt

# Utilizado en las capas de Autoencoder
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, Input, BatchNormalization, Dropout, Dense
from tensorflow.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.keras import backend as K

# Utilizado en funcion de entrenamiento
from sklearn.model_selection import train_test_split


class Autoencoder:
    """Resumen de la clase
    Esta clase se compone de varias funciones para la creación, entrenamiento y muestra de resultados de un modelo Autoencoder
    dicho modelo se basa en codificar y descodificar los datos del conjunto de datos para obtener un conjunto de neuronas que 
    permiten obtener un conjunto de datos mayoritariamente equivalente al conjunto de datos de entrada. Nuestro Autoencoder se
    compone de unas capas Dense con Tied Weights, donde los peso son los que los pesos que componen las capas del Encoder
    pero estos pesos están Transpuestos.

    Hay funciones que nos permiten realizar un proceso de Fine Tuning, que se basa en eliminar la capa decoder para añadir
    una capa softmax y cambiar el modelo de entrenamiento de nuestro modelo.

    Hay funciones que nos permiten mostrar los valores de funcion de perdida y métricas utilizadas en las etapas de entrenamiento 
    tanto en el modelo Autoencoder como en el modelo Fine tuning.
    

    Atributos:
        _input_shape: Entero que indica el valor inicial del conjunto de datos de entrada
        _layer_sizes: Array de enteros que indica la cantidad de capas del Encoder y Decoder
        _num_layers: Entero que indica el numero de capas de _layer_sizes
        _latent_space: Entero que indica la dimensionalidad de la capa bottleneck
        _dropout_value: Decimal que indica el valor dropout de las capas
        _activation_func: String que indica el tipo de función de activación del modelo

        _encoder: Variable que almacena las capas del encoder del modelo
        _decoder: Variable que almacena las capas del decoder del modelo

        _model_input: Variable que almacena la capa input del modelo
        _encoder_output: Variable que almacena la capa output del decoder

        _model: Variable que almacena toda la información (capas, entrenamiento...etc) del modelo
        _model_fine_tuning: Variable que almacena toda la información (capas, entrenamiento...etc) del modelo una vez realizada la etapa de Fine Tuning

        _autoencoder_train: Variable que almacena toda la información del entrenamiento del Autoencoder
        _autoencoder_fine_tuned_train: Variable que almacena toda la información del entrenamiento del Autoencoder una vez realizada la etapa de Fine Tuning

    """

    def __init__(self,
        input_shape,
        layer_sizes,
        latent_space,
        drop_out_value,
        activation_func):
        """ 
        Inicialización de las variables de la clase Autoencoder
        
        :param input_shape: Numero de atributos que tiene el conjunto de entrada
        :param layer_sizes: Numero de capas que tendra el Autoencoder
        :param latent_space: Numero de neuronas que tendra la capa bottlenekc
        :param drop_out_value: Valor dropout definido en las capas dense
        :param activation_func: Funcion de activacion usada en el Autoencoder
        :return: Creamos el Autoencoder para poder ser entrenado
        """

        self._input_shape = input_shape # Tamaño de entrada del modelo
        self._layer_sizes = layer_sizes
        self._num_layers = len(layer_sizes)
        self._latent_space = latent_space
        self._dropout_value = drop_out_value
        self._activation_func = activation_func

        self._encoder = None
        self._decoder = None

        self._model_input = None
        self._encoder_output = None

        self._model = None
        self._model_fine_tuning = None
        
        self._autoencoder_train = None
        self._autoencoder_fine_tuned_train = None

        self._build()


    def _build(self):
        """
        Construccion del Encoder, Decoder y Autoencoder
        
        :return: Modelo Autoencoder creado
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        """
        Creacion el modelo Encoder
        
        :return: Modelo Encoder inicializado con sus capas correspondientes
        """

        encoder_input = self._add_encoder_input()
        encoder_layers = self._add_layers_encoder(encoder_input)
        encoder_output = self._add_encoder_output(encoder_layers)

        self._model_input = encoder_input

        self._encoder = Model(encoder_input, encoder_output, name = "Encoder_Layer")

    def _build_decoder(self):
        """
        Creacion el modelo Decoder
        
        :return: Modelo Decoder inicializado con sus capas correspondientes
        """

        decoder_input = self._add_decoder_input()
        decoder_layers = self._add_layers_decoder(decoder_input)
        decoder_output = self._add_decoder_output(decoder_layers)

        self._decoder = Model(decoder_input, decoder_output, name = "Decoder_Layer")

    def _build_autoencoder(self):
        """
        Creacion del modelo Autoencoder a partir del Encoder y Decoder
        
        :return: Modelo Autoencoder inicializado
        """

        model_input = self._model_input
        model_output = self._decoder(self._encoder(model_input))

        self._model = Model(model_input, model_output, name = "Autoencoder_Model")

    def _add_encoder_input(self):
        """
        Devolvemos un modelo con una capa Input inicial para el modelo Encoder
        
        :return: Devolvemos un modelo x con una capa Input inicial
        """

        x =  Input(shape = (self._input_shape,),
            dtype = tf.float32, 
            name = "Encoder_Input")

        return x

    def _add_decoder_input(self):
        """
        Devolvemos un modelo con una capa Input inicial para el modelo Decoder
        
        :return: Devolvemos un modelo x con una capa Input inicial
        """

        x = Input(shape=(self._latent_space,),
            dtype = tf.float32, 
            name = "Decoder_Input")

        return x

    def _add_encoder_output(self,x):
        """
        Devolvemos un modelo anyadiendo una capa Dense, BN y Dropout
        
        :param x: Modelo que representa el Encoder
        :return: Devolvemos un modelo x anyadiendo la capa final del Encoder
        """

        input_shape_layer = (self._layer_sizes[-1],) if len(self._layer_sizes) != 0 else (self._input_shape,)
        x = Dense(self._latent_space,
            activation = self._activation_func, 
            input_shape = input_shape_layer,
            kernel_initializer = initializers.he_normal,
            kernel_regularizer = tensorflow.keras.regularizers.L2(0.01),
            use_bias = True)(x)
        x = BatchNormalization()(x)
        x = Dropout(self._dropout_value)(x)

        return x

    def _add_decoder_output(self,x):
        """
        Devolvemos un modelo anyadiendo una ultima capa al modelo Decoder
        
        :param x: Modelo que representa el Encoder
        :return: Devolvemos un modelo x anyadiendo la capa final del Decoder
        """

        x = DenseTied(self._input_shape, 
            activation = "sigmoid",
            use_bias = True,
            tied_to = self._encoder.layers[1])(x)

        return x

    def _add_layers_encoder(self, encoder_input):
        """
        Devolvemos un modelo anyadiendo un conjunto de capas al Encoder segun el valor _num_layers
        El conjunto de capas se compone de un capa Dense, BN y Dropout
        
        :param encoder_input: Modelo que representa el Input del Encoder
        :return: Devolvemos un modelo x con un numero concreto de capas
        """

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

    def _add_layers_decoder(self, decoder_input):
        """
        Devolvemos un modelo anyadiendo un conjunto de capas al Decoder segun el valor _num_layers
        El conjunto de capas se compone de un capa Dense con pesos enlazados, BN y Dropout
        
        :param decoder_input: Modelo que representa el Input del Decoder
        :return: Devolvemos un modelo x con un numero concreto de capas
        """

        x = decoder_input
        for layer_index in reversed(range(1, self._num_layers + 1)):
            input_dim = self._layer_sizes[layer_index - 1]
            x = self.__add_decoded__dense(input_dim,
                self._encoder.layers[1 + layer_index * 3],
                self._activation_func,
                x)

            x = BatchNormalization()(x)
            x = Dropout(self._dropout_value)(x)

        return x

    def __add_encoded_dense(self, node_size, input_dim, activation_func, x):
        """ 
        Creamos la capa Dense y la anyadimos al modelo

        :param node_size: Variable que indica el tamaño de neuronas de la capa Dense
        :param input_dim: Variable que indica el tamaño de la capa anterior
        :param activation_func: Variable que indica la funcion de activacion de la capa Dense
        :param x: Modelo que representa el Encoder
        :return: Devolvemos un modelo x con una capa Dense
        """

        x = Dense(node_size, 
            activation = activation_func,
            input_shape = (input_dim,),
            kernel_initializer = initializers.he_normal,
            use_bias = True)(x)

        return x

    def __add_decoded__dense(self, input_dim, encoder_layer, activation_func, x):
        """ 
        Creamos la capa Dense con pesos enlazados de las capas del Encoder y la anyadimos al modelo

        :param input_dim: Variable que indica el tamaño de la capa anterior
        :param encoder_layer: Variable que representa una capa Dense correspondiente al Decoder
        :param activation_func: Variable que indica la funcion de activacion de la capa Dense
        :param x: Modelo que representa el Encoder
        :return: Devolvemos un modelo x con una capa Dense 
        """

        x = DenseTied(input_dim, 
            activation = activation_func,
            tied_to = encoder_layer,
            use_bias = True)(x)
                      
        return x

    def get_autoencoder_model(self):
        """
        Devolvemos el modelo _model
        
        :return: Devolvemos un modelo _model previamente creado
        """

        return self._model


    def fine_tuning(self, number_outputs):
        """
        Realizamos un cambio Fine Tuning al Autoencoder, se elimina la capa Decoder del Autoencoder
        y se le anyade una capa Softmax que predice la clase que pertenece cada dato del conjunto de datos

        :param number_outputs: Numero de clases que compone el conjunto de datos
        """

        self._model_fine_tuning = Sequential()
        self._model_fine_tuning.add(Input(shape = (self._input_shape,),
                                    dtype = tf.float32 ))
        
        for layer in self._encoder.layers:
            self._model_fine_tuning.add(layer)

        self._model_fine_tuning.add(Dense(number_outputs, activation = 'softmax'))

    def set_trainable(self, trainable):
        """
        Cambiamos la variable trainable de cada una de las capas del Autoencoder con Fine Tuning para
        que esas capas entrenen o no entrenen

        :param trainable: Booleano que simboliza si las capas entrenan o no entrenan
        """

        for layer in self._model_fine_tuning.layers:
            layer.trainable = trainable


    def summary(self):
        """
        :return: Muestra por pantalla con que capas esta formado los modelos creados
        y el numero de parametros que entrena
        """

        self._encoder.summary()
        self._decoder.summary()
        self._model.summary()
    

    def summary_fine_tuning(self):
        """
        :return: Muestra por pantalla con que capas esta formado los modelos creados
        y el numero de parametros que entrena
        """

        self._model_fine_tuning.summary()

    def compile(self, learning_rate = 0.0001):
        """
        Compilamos nuestro modelo para que pueda entrenar, utilizando Adam como optimizador 
        y MAE como funcion de perdida

        :param learning_rate: Es el valor de aprendizaje utilizado en el entrenamiento
        :return: _model ya es un modelo preparado para ser entrenado
        """

        optimizer = Adam(learning_rate = learning_rate)
        loss = tensorflow.keras.losses.MeanAbsoluteError()
        self._model.compile(optimizer = optimizer, loss = loss,
            metrics = [tensorflow.keras.metrics.MeanAbsoluteError()])

    def compile_fine_tuning(self, learning_rate = 0.0001):
        """
        Compilamos nuestro modelo para que pueda entrenar, utilizando Adam como optimizador 
        y Sparse Categorical Crossentropy como funcion de perdida

        :param learning_rate: Es un valor entero que identifica aprendizaje utilizado en el entrenamiento
        :return: _model_fine_tuning ya es un modelo preparado para ser entrenado
        """

        optimizer = Adam(learning_rate = learning_rate)
        loss = tensorflow.keras.losses.SparseCategoricalCrossentropy(reduction = "sum")
        self._model_fine_tuning.compile(optimizer = optimizer, loss = loss,
            metrics=[tensorflow.keras.metrics.SparseCategoricalAccuracy()])

    def train(self, x_train, batch_size, num_epochs, verbose_mode):
        """
        Entrenamiento del autoencoder,se divide los datos x_train en datos de entrenamiento y datos de validación, 
        se utiliza la variable callback para parar al Autoencoder si no mejora sus valores.
        
        :param x_train: Array de 2 dimensiones que contiene el conjunto de datos para entrenar
        :param batch_size: Valor entero que representa el batch size del entrenamiento
        :param num_epochs: Valor entero que representa el numero de epocas de un entrenamiento
        :param verbose_mode: Valor entero que puede tomar los valores 0,1 y 2. Segun el valor que se le asigne se vera
        los datos correspondientes al entrenamiento de una manera u otra
        :return: El modelo _autoencoder_train entrena correctamente y los pesos y bias de sus capas estan ajustados
        segun el tipo de problema que ha tenido en la entrada
        """
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

    def train_fine_tuning(self, x_train, y_train, batch_size, num_epochs, verbose_mode):
        """
        Entrenamiento del autoencoder tras aplicar Fine Tuning, se divide los datos x_train e y_train en datos de entrenamiento
        y datos de validación, se utiliza la variable callback para parar al Autoencoder si no mejora sus valores.
        
        :param x_train: Array de 2 dimensiones que contiene el conjunto de datos para entrenar
        :param y_train: Array de 1 dimension de valores enteros que contiene la clase que corresponde a cada uno de los
        diferentes datos de x_train
        :param batch_size: Valor entero que representa el batch size del entrenamiento
        :param num_epochs: Valor entero que representa el numero de epocas de un entrenamiento
        :param verbose_mode: Valor entero que puede tomar los valores 0,1 y 2. Segun el valor que se le asigne se vera
        los datos correspondientes al entrenamiento de una manera u otra
        :return: El modelo _autoencoder_fine_tuned_train entrena correctamente y los pesos y bias de sus capas estan ajustados
        segun el tipo de problema que ha tenido en la entrada
        """

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

    def predict(self, data):
        """
        Prediccion del _model Autoencoder del array data

        :param data: Array de 2 dimensiones de datos
        :return: Devuelve un array de tamaño equivalente a data el cual contiene
        los valores que segun el Autoencoder _model ha estimado segun el entrenamiento
        """
        return self._model.predict(data)
    
    def predict_reduced(self, data):
        """
        Prediccion del _encoder Encoder del array data

        :param data: Array de 2 dimensiones
        :return: Devuelve un array de tamaño reducido segun el valor establecido en la
        capa dense bottleneck, los valores son los que estima el autoencoder con el entrenamiento
        """
        return self._encoder.predict(data)


    def predict_fine_tuning(self, data, verbose_mode):
        """
        Prediccion del modelo Autoencoder _model_fine_tuning, donde para cada valor perteneciente al
        array data se le predice la posible clase que deberia ser. En primer lugar obtenemos la probabilidad
        de pertenecer a cada una de las clases y la probabilidad que mayor valor tiene es la clase que debe
        pertenecer el valor

        :param data: Array de 2 dimensiones
        :param verbose_mode: Variable que puede tomar los valores 0,1,2. Depende de que valor se mostraran
        unos valores por pantalla u otros de la prediccion
        :return: Devuelve un array de 1 dimension result que contiene la clase que pertenece cada valor del
        array data
        """
        
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
    
    def predict_proba(self, data, verbose_mode):
        """
        Prediccion del modelo Autoencoder _model_fine_tuning, donde para cada valor perteneciente al
        array data se indica la probabilidad de pertenecer a cada una de las clases del conjunto de datos

        :param data: Array de 2 dimensiones
        :param verbose_mode: Variable que puede tomar los valores 0,1,2. Depende de que valor se mostraran
        unos valores por pantalla u otros de la prediccion
        :return: Devuelve un array de 1 dimension result que contiene probabilidad de pertenecer a cada una de
        las clases que corresponden el problema
        """

        return self._model_fine_tuning.predict(data, verbose = verbose_mode)

    def return_reduce_attribute(self, x_data):
        """
        Nos quedamos solo con las n-1 layers del modelo fine tuning ya entrenado, es decir, eliminamos
        la capa softmax, tras esto predecimos dando como parametros la variable x_data obteniendo un
        conjunto de datos reducido

        :param data: Array de 2 dimensiones
        :return: Devuelve un conjunto de datos reducido segun el modelo new_model
        """

        layers = self._model_fine_tuning.layers

        new_model = tensorflow.keras.models.Sequential(layers[:-1])

        return new_model.predict(x_data)
    
    def get_encoder_weights(self, layer_index):
        """
        Funcion get para devolver valores y evaluarlos

        :param layer_index: Variable entera que indica el indice de la capa dense que queremos acceder 
        :return: Devuelve los pesos correspondientes a la i-esima capa dense
        """
        return self._encoder.layers[1 +  3 * layer_index].get_weights()

    def obtain_history(self, autoencoder_type, problem_type):
        """
        Muestra por pantalla una grafica de los valores obtenidos durante el entrenamiento el
        autoencoder por cada uno de los epochs realizados

        :param autoencoder_type: String, si el valor es "fine_tuning" se utilizaran los valores obtenidos
        por el autoencoder fine tuning en la grafica
        :param problem_type: String, se utiliza para guardar el nombre del archivo correspondiente al problema
        """

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
            plt.xlabel("Epochs")
            plt.ylabel(metric_name)
            plt.legend()
            plt.savefig("../results/train_metrics/" + metric_name + "_" + autoencoder_type + problem_type + '.png', bbox_inches='tight')

class DenseTied(Layer):
    """Resumen de la clase
    Esta clase se compone de las funciones y atributos para poder enlazar los pesos de las capas dense del decoder a los pesos de las capas
    dense de los encoders
    Clase obtenida a partir del articulo: https://medium.com/@lmayrandprovencher/building-an-autoencoder-with-tied-weights-in-keras-c4a559c529a2
    Se han hecho varias modificaciones debido a que no funcionaba correctamente por la version de Keras

    Atributos:
        units: Numero de neuronas que va a utilizar el decoder
        activation: Funcion de activacion de las neuronas
        use_bias: Bool, si se quiere utilizar bias True, "False" en caso contrario
        bias_initializer: Dependiendo del valor indicado se inicializara el bias de una manera u otra
        mirar documentacion en Keras para saber sus valores
        bias_regularizer: Segun su valor se utilizara un regularizador para los valores bias
        mirar documentacion en Keras para saber sus valores
        bias_constraint: Funcion constrain aplicado al bias, mirar documentacion en Keras
        para saber sus valores 
        tied_to: Modelo al cual esta el decoder enlazado
    """

    def __init__(self, units,
                activation=None,
                use_bias=True,
                bias_initializer='zeros',
                bias_regularizer=None,
                bias_constraint=None,
                tied_to=None,
                **kwargs):
        
        """ 
        Inicialización de las variables de la clase Autoencoder
        
        :param units: Numero de neuronas que va a utilizar el decoder
        :param activation: Funcion de activacion de las neuronas
        :param use_bias: Bool, si se quiere utilizar bias True, "False" en caso contrario
        :param bias_initializer: Dependiendo del valor indicado se inicializara el bias de una manera u otra
        mirar documentacion en Keras para saber sus valores
        :param bias_regularizer: Segun su valor se utilizara un regularizador para los valores bias
        mirar documentacion en Keras para saber sus valores
        :param bias_constraint: Funcion constrain aplicado al bias, mirar documentacion en Keras
        para saber sus valores 
        :param tied_to: Modelo al cual esta el decoder enlazado
        """
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


    def build(self, input_shape):
        """
        Funcion que asigna los pesos y bias correspondientes a la capa dense ya construida,
        si la capa dense tiene una capa enlazada, se le asignaran los pesos de esa capa.
        En caso contrario se le asignaran otros segun como esten establecidos en el kerner de
        la otra capa Dense. Tambien el valor de los bias se inicializa o se tiene en cuenta segun
        la variable use_bias o los parametros de inicializacion

        :param input_shape: Dimension de los datos de entrada
        """
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
        """
        :param input_shape: Dimension de los datos de entrada
        :return: Devuelve una tupla con el mismo tamaño que la variable output_shape, que esta
        depende del primer valor del array input_shape, que es el tamaño de filas que contiene
        nuestro conjunto de datos
        """
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1] == self.units
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        """
        Funcion que se ejecuta cuando se entrenan los modelos

        :param inputs: Conjunto de datos de entrada
        :return: Devuelve la salidad de los datos depues del multiplicar los datos de entrada
        con la matriz trapuesta de los pesos enlazados, sumandolo al final con la matriz de los bias
        """
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output
