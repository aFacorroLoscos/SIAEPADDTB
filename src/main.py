# Bibliotecas generales
import os
import sys
import numpy as np
import keras
import pandas as pd

# Para calcular el tiempo de aprendizaje
import time

# Biblioteca para Kfold, split de datos y labelice de las clases
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize

# Clases propias
from aeClass import Autoencoder
from loadMedicData import Datasets
from MAUC import MAUC 
import plot

# Shallow models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.base import clone

# Metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_curve, auc, cohen_kappa_score

# Hyper parameters optimization
from skopt import BayesSearchCV 
from scikeras.wrappers import KerasRegressor

import numpy as np


# Variables GLOBALES PARA ENTRENAR EL AUTOENCODER
LEARNINGRATE = 0.004
BATCHSIZE = 32
DROPOUTVALUE = 0.2
EPOCHS = 50

DEST_FOLDER = "CN_MCI_AD"

# Diccionarios
SHALLOWMODELDICT = {
    "randomforest": RandomForestClassifier()
}

def read_file(file_name_path):
    """ 
    Lee un fichero segun file_name_path y guarda todo su contenido en un array

    :param file_name_path: String que es el path del fichero a leer
    :return: Devuelve un array de palabras
    """
    file_obj = open(file_name_path, "r")
    words = file_obj.read().splitlines()
    file_obj.close()
    return words


def one_hot(array, num_classes):
    """
    
    :param array: Array de valores de entrada
    :param num_classes: Numero de clases del array
    :return: Devolvemos un array tras ejecutar algoritmo One Hot, donde el numero
    de columnas es equivalente al num_classes y la columna correspondiente a la clase
    del valor es un 1
    """
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])

def train_autoencoder(x_train, y_train, learning_rate, 
    batch_size, epochs, dropout,
    size_layers_used, latent_space, activation_func,
    number_classes, autoencoder_model, 
    verbose, verbose_train):
    
    """
    Creamos un autoencoder base, se entrena, se realiza una etapa fine tuning y se entrena dicha etapa

    :param x_train: Contiene las variables de cada observacion
    :param y_train: Contiene la clase a la que pertenece cada observacion
    :param learning_rate: Ratio de aprendizaje del autoencoder
    :param batch_size: Numero de batch size para el entrenamiento
    :param epochs: Numero de epocas del entrenamiento
    :param dropout: Valor dropout para las capas del autoencoder
    :param size_layers_used: Array que contiene el numero de cada una de las capas
    :para latent space: Numero de neuronas de la capa bottleneck
    :param activation_func: Funcion de activacion de las capas
    :param number_classes: Numero de clases que contiene el problema
    :param autoencoder_model: Bool, False si no quieres utilizar un autoencoder, True si quieres utilizarlo. 
    Se usa para crear una red neuronal o un autoencoder
    :param verbose: True si quieres guardar los datos del historial de entrenamiento, False si no quieres
    :param verbose_train: Dependiendo del valor muestra un output diferente en el entrenamiento
    :return: Devuelve un autoencoder entrenado con la etapa fine tuning realizada y entrenada
    """

    # Creamos el autoencoder
    autoencoder = Autoencoder(
        input_shape = (np.shape(x_train)[1]),
        layer_sizes = size_layers_used,
        latent_space = latent_space,
        drop_out_value = dropout,
        activation_func = activation_func,
    )
    
    if verbose:
        autoencoder.summary()

    if(autoencoder_model):
        # Entrenamos el autoencoder
        autoencoder.compile(learning_rate)
        autoencoder.train(x_train, batch_size, epochs, verbose_train)

        if verbose:
            autoencoder.obtain_history("normal", "_" + DEST_FOLDER + "_")

    # Entrenamos el autoencoder para que pueda
    # clasificar mediante fine tuning
    autoencoder.fine_tuning(number_classes)

    if verbose:
        autoencoder.summary_fine_tuning()

    autoencoder.compile_fine_tuning(learning_rate/10)  
    autoencoder.set_trainable(True)
    autoencoder.train_fine_tuning(x_train, y_train, batch_size, epochs, verbose_train)

    if verbose:
        autoencoder.obtain_history("fine_tuning", "_" + DEST_FOLDER + "_")

    return autoencoder

def kFold_cross_validation_DLmodel(x_train, y_train, lr, 
                                    batch_size, epochs, dropout_value,
                                    size_layers_used, latent_space,
                                    activation_func, number_classes, autoencoder_model):
    """
    Mostramos por pantalla las metricas obtenidas a partir de un modelo si utilizamos
    el proceso K fold cross validation

    :param x_train: Contiene las variables de cada observacion
    :param y_train: Contiene la clase a la que pertenece cada observacion
    :param lr: Ratio de aprendizaje del autoencoder
    :param batch_size: Numero de batch size para el entrenamiento
    :param epochs: Numero de epocas del entrenamiento
    :param dropout_value: Valor dropout para las capas del autoencoder
    :param size_layers_used: Array que contiene el numero de cada una de las capas
    :para latent space: Numero de neuronas de la capa bottleneck
    :param activation_func: Funcion de activacion de las capas
    :param number_classes: Numero de clases que contiene el problema
    :param autoencoder_model: Bool, False si no quieres utilizar un autoencoder, True si quieres utilizarlo. 
    Se usa para crear una red neuronal o un autoencoder
    """
    
    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(x_train, y_train)
    # Metricas a evaluar
    accuracys = []
    precisions = []
    recalls = []
    F1scores = []  
    # Proceso K fold cross validation
    for k, (train, test) in enumerate(kfold):
        autoencoder = train_autoencoder(x_train[train], y_train[train], 
            lr, batch_size, epochs, dropout_value, 
            size_layers_used, latent_space, activation_func, 
            number_classes, autoencoder_model, 
            False, 0)
        
        metrics = get_metrics(autoencoder.predict_fine_tuning(x_train[test], 0), y_train[test])
        accuracys.append(metrics[0])
        precisions.append(metrics[1])
        recalls.append(metrics[2])
        F1scores.append(metrics[3])

        print('Fold: %2d, Training/Test Split Distribution: %s - %s, Accuracy: %.3f, Precision: %.3f, Recall: %.3f, F1Score: %.3f' % (k+1, 
                                                                                        np.shape(y_train[train])[0], np.shape(y_train[test])[0],
                                                                                        metrics[0], metrics[1], metrics[2], metrics[3]), flush=True)
    # Mostramos valores finales
    model_name = "Autoencoder" if autoencoder_model else "Neural Network"
    print("Result obtained using " + model_name + " model:")
    print('Cross-Validation accuracy: %.3f +/- %.3f' %(np.mean(accuracys), np.std(accuracys)))
    print('Cross-Validation precision: %.3f +/- %.3f' %(np.mean(precisions), np.std(precisions)))
    print('Cross-Validation recall: %.3f +/- %.3f' %(np.mean(recalls), np.std(recalls)))
    print('Cross-Validation meanF1: %.3f +/- %.3f\n ' %(np.mean(F1scores), np.std(F1scores)))


def kFold_cross_validation_shallow(model, x_values, y_values, model_name):
    """
    Mostramos por pantalla las metricas obtenidas a partir de un modelo si utilizamos
    el proceso K fold cross validation

    :param model: Modelo de aprendizaje tradicional
    :param x_values: Contiene las variables de cada observacion
    :param y_values: Contiene la clase a la que pertenece cada observacion
    :param model_name: String, contiene el nombre del modelo
    """

    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(x_values, y_values)
    # Metricas a evaluar
    accuracys = []
    precisions = []
    recalls = []
    meansF1 = [] 
    # Proceso K fold cross validation
    for k, (train, test) in enumerate(kfold):
        
        new_model = clone(model).fit(x_values[train], y_values[train])

        metrics = get_metrics(y_values[test], new_model.predict(x_values[test]))
        
        accuracys.append(metrics[0])
        precisions.append(metrics[1])
        recalls.append(metrics[2])
        meansF1.append(metrics[3])
        
        print('Fold: %2d, Training/Test Split Distribution: %s - %s, Accuracy: %.3f' % (k+1, 
            np.shape(y_values[train])[0],
            np.shape(y_values[test])[0],
            metrics[0]))
    # Mostramos valores finales
    print("Result obtained using " + model_name + " model:")
    print('Cross-Validation accuracy: %.3f +/- %.3f' %(np.mean(accuracys), np.std(accuracys)))
    print('Cross-Validation precision: %.3f +/- %.3f' %(np.mean(precisions), np.std(precisions)))
    print('Cross-Validation recall: %.3f +/- %.3f' %(np.mean(recalls), np.std(recalls)))
    print('Cross-Validation meanF1: %.3f +/- %.3f\n ' %(np.mean(meansF1), np.std(meansF1)))

def hyper_parameters_autoencoder(x_train, y_train, layer_sizes, latent_space, encoder_act_func):

    """
    Obtencion de los mejores hiper parametros del Autoencoder, se realiza un grid search, es decir,
    se busca por todas las combinaciones de parametros definidos segun n^i, siendo i el numero de 
    parametros diferentes a tener en cuenta.
    Muestra por pantalla que combinacion de parametros es la mejor

    :param x_train: Contiene las variables de cada observacion
    :param y_train: Contiene la clase a la que pertenece cada observacion
    :param layer_sizes: Array que contiene el numero de cada una de las capas
    :param latent_space: Numero de neuronas de la capa bottleneck
    :param encoder_act_func: Funcion de activacion del autoencoder
    """

    # Definimos los parametros para la busqueda
    learningrate = [0.01, 0.001, 0.0001]
    learningrate_fine_tuning = [0.001, 0.0001, 0.00001]
    batch_size = [32]
    dropout_values = [0.3,0.4]
    epochs = [15,20,25]

    # Algoritmo Grid Search propio
    best_precision = 0
    best_hyper_parameters = [0, 0, 0, 0, 0]
    for j in range(len(batch_size)):
        for l in range(len(epochs)):
            for i in range(len(learningrate)):
                for k in range(len(dropout_values)):
                    print("Using: Dropout " + str(dropout_values[k]) + " , LRate: " + str(learningrate[i]) + ", Epochs: " + str(epochs[l]) + " , BatchSize: " + str(batch_size[j]), flush=True)
                    current_precision = kFold_cross_validation_DLmodel(x_train, y_train, 
                        learningrate[i], batch_size[j], 
                        epochs[l], dropout_values[k],
                        layer_sizes, latent_space,
                        encoder_act_func, 3, False)
                    if current_precision > best_precision:
                        best_hyper_parameters[0] = learningrate[i]
                        best_hyper_parameters[1] = learningrate_fine_tuning[i]
                        best_hyper_parameters[2] = batch_size[j]
                        best_hyper_parameters[3] = dropout_values[k]
                        best_hyper_parameters[4] = epochs[l]
                        bestPrecision = current_precision

    # Resultados
    print("Best F1 Score: " + str(best_precision))
    print("Best Hyperparameters: ")
    print("Learning Rate: " + str(best_hyper_parameters[0]))
    print("Learning Rate FT: " + str(best_hyper_parameters[1]))
    print("Batch Size: " + str(best_hyper_parameters[2]))
    print("Dropout: " + str(best_hyper_parameters[3]))
    print("Epochs: " + str(best_hyper_parameters[4]))

def create_autoencoder_hyper_par(dropout_value):
    """
    Creamos un autoencoder para que la funcion KerasRegressor tenga un modelo y pueda 
    realizar la busqueda bayesiana, los parametros de esta funcion se utilizan como espacio de 
    busqueda en la busqueda bayesiana

    :param dropout_value: Valor dropout para las capas del autoencoder
    :return: Devuelve un autoencoder para la busqueda bayesiana
    """

    autoencoder = Autoencoder(
        input_shape = 740,
        layer_sizes = [600,300,200],
        latent_space = 150,
        drop_out_value = dropout_value,
        activation_func = "relu",
    )

    autoencoder_model = autoencoder.get_autoencoder_model()

    return autoencoder_model

def hyper_parameters_optimization(x_train):
    """
    Obtencion de los mejores hiper parametros del Autoencoder, se realiza una búsqueda bayesiana
    Muestra por pantalla que combinacion de parametros es la mejor y que valores ha generado

    :param x_train: Contiene las variables de cada observacion
    """

    # Callback y kfold para la busqueda bayesiana
    callback =  keras.callbacks.EarlyStopping(monitor = "loss", mode = "min", verbose = 1, patience = 10, min_delta = 0.01)
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = True)
    # Parametros y que espacio de busqueda tendran
    search_spaces = {
        "optimizer__learning_rate" : (1e-3, 0.1, "log-uniform") ,
        "batch_size" : [16,32,64],
        "model__dropout_value" : [0.2,0.3,0.4]
    }
    # Busqueda bayesiana
    autoencoder_model = KerasRegressor(create_autoencoder_hyper_par, loss = "mean_absolute_error", optimizer = "adam", epochs = 50, callbacks = [callback], verbose = 0, random_state = 42)
    opt = BayesSearchCV(
        estimator = autoencoder_model,
        search_spaces = search_spaces,
        n_jobs = -1,
        cv = kfold,
        verbose = 0
    )
    opt.fit(x_train, x_train)
    # Resultados
    print("Best val. score: %s" % opt.best_score_)
    print("Best params obtained: %s" % str(opt.best_params_))


def get_metrics(y_test, y_pred):
    """
    Obtenemos las metricas de precision, recall, f1 score y accuracy de las variables y_test e y_pred
    
    :param y_test: Contiene la clase a la que pertenece cada observacion
    :param y_pred: Contiene la clase predicha de cada observacion
    :return: Devuelve un array de metricas 
    """
    clasifier_metrics = []
    clasifier_metrics.append(accuracy_score(y_test, y_pred))
    clasifier_metrics.append(precision_score(y_test, y_pred, average='weighted', zero_division = 0))
    clasifier_metrics.append(recall_score(y_test, y_pred, average='weighted', zero_division = 0))
    clasifier_metrics.append(f1_score(y_test, y_pred, average='weighted'))
    return clasifier_metrics

def show_metrics(metrics):
    """
    Mostramos las metricas de metrics por pantalla

    :param metrics: Array que contiene las metricas de accuracy, precision, recall y f1 score
    """
    print("Accuracy: " + str(round(metrics[0],3)))
    print("Precision: " + str(round(metrics[1],3)))
    print("Recall: " + str(round(metrics[2],3)))
    print("F1 Score: " + str(round(metrics[3],3)))

def get_metrics_model(model, x_test, y_test, model_name):
    """
    Realizamos una prediccion de los valores x_test y se evaluan con los valores y_test obteniendo unas metricas
    las cuales mostramos por pantalla y devolvemos 

    :param model: Modelo de aprendizaje
    :param x_test: Contiene las variables de cada observacion
    :param y_test: Contiene la clase a la que pertenece cada observacion
    :param model_name: Nombre del modelo
    :return: Devuelve un array de metricas obtenidas a partir del modelo modelo
    """
    
    metrics = get_metrics(y_test, model.predict(x_test))
    print("Metrics obtained " + model_name + ":")
    show_metrics(metrics)
    return metrics

def train_show_time(model, x_train, y_train, model_name):
    """
    Mostramos el tiempo de entrenamiento del modelo model

    :param model: Modelo de aprendizaje
    :param x_train: Contiene las variables de cada observacion
    :param y_train: Contiene la clase a la que pertenece cada observacion
    :param model_name: Nombre del modelo
    """
    t0  = time.time()
    model.fit(x_train, y_train)
    print("Training Time from " + model_name + ":", time.time()-t0)

def got_roc_values(x_data, y_data):
    """
    Generamos los valores false positive rate y true positive rate mediante roc_curve
    Despues generamos el valor roc_auc correspondiente

    :param x_data: Contiene las variables de cada observacion
    :param y_data: Contiene la clase a la que pertenece cada observacion
    :return: Devolvemos tres valores, fpr, tpr y roc auc para generar las grafica ROC Curve
    """

    fpr, tpr, thresholds = roc_curve(x_data, y_data)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def got_roc_values_multiclass(y_test, y_scores, num_classes):
    """
    Generamos los valores false positive rate y true positive rate mediante roc_curve
    Despues generamos el valor roc_auc correspondiente

    :param y_test: Contiene la clase a la que pertenece cada observacion
    :param y_scores: Array que contiene la probabilidad de pertenecer a la clase de cada observacion
    :param num_classes: Numero de clases del problema
    :return: Devolvemos tres valores, fpr, tpr y roc auc para generar las grafica ROC Curve
    """

    y_test_onehot = one_hot(y_test, num_classes)
    fpr, tpr, _ = roc_curve(y_test_onehot.ravel(), y_scores.ravel())
    return fpr, tpr, auc(fpr, tpr)

def got_zip_class_prob(class_list, prob_list):
    """
    
    :param class_list: Lista de clases del conjunto de datos
    :param prob_list: Lista de probabilidad de pertenecer a la clase del conjunto de datos
    :return: Devolvemos un array que contiene array de tuplas combinadas entre class_list y prob_list
    mediante la funcion zip
    """
    zip_true_label_probs = []
    for x in range(len(class_list)):
        zip_true_label_probs += [(class_list[x], list(prob_list[x]))]
    
    return zip_true_label_probs


def compare_models_bar_graph(x_eval, reduced_eval, y_eval, 
        autoencoder, trained_model_dict, trained_model_dict_reduced):
    """
    Comparacion de modelos mediante grafico de barras

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param trained_model_dict_reduced:  Modelos de aprendizaje tradicional entrenados con un conjunto de datos de dimensionalidad reducida
    """

    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_bar_graph(x_eval, reduced_eval, y_eval, 
            autoencoder, model, model_reduce,
            name_1)

def compare_models_roc_curve(x_eval, reduced_eval, y_eval, 
        autoencoder, trained_model_dict, trained_model_dict_reduced, 
        number_classes):
    """
    Comparacion de modelos mediante curvas ROC

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param trained_model_dict_reduced:  Modelos de aprendizaje tradicional entrenados con un conjunto de datos de dimensionalidad reducida
    :param number_classes: Numero de clases que contiene los datos 
    """
    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_roc_curve( x_eval, reduced_eval, y_eval, 
            autoencoder, model, model_reduce,
            name_1,number_classes)

def compare_models_MAUC(x_eval, reduced_eval, y_eval, 
        autoencoder, trained_model_dict, trained_model_dict_reduced, 
        number_classes):
    """
    Comparacion de modelos mediante valores MAUC

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param trained_model_dict_reduced:  Modelos de aprendizaje tradicional entrenados con un conjunto de datos de dimensionalidad reducida
    :param number_classes: Numero de clases que contiene los datos 
    """
    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_mauc(x_eval, reduced_eval, y_eval, 
            autoencoder, model, model_reduce,
            name_1,number_classes)
    

def show_time(x_train, y_train, trained_model_dict, using_DL):
    """
    Muestra el tiempo de ejecucion del entrenamiento de cada uno de los modelos

    :param x_train: Contiene las variables de cada observacion
    :param y_train: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    """
    for name, model in trained_model_dict.items():
        train_show_time(model, x_train, y_train, name + using_DL)

def show_result(x_eval, y_eval, trained_model_dict, using_DL, number_classes):
    """
    Evaluacion mediante matriz de confusion, grafica de barras y curvas roc
    entre los diferentes modelos entrenados

    :param x_eval: Contiene las variables de cada observacion
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    :param number_classes: Numero de clases que contiene los datos 
    """

    plot_matrix(x_eval, y_eval, trained_model_dict, using_DL, number_classes)
    show_result_bar_graph(x_eval, y_eval, trained_model_dict, using_DL, number_classes)
    show_result_roc_curves(x_eval, y_eval, trained_model_dict, using_DL, number_classes)

def plot_matrix(x_eval, y_eval, trained_model_dict, using_DL, number_classes):
    """
    Generacion de matrices de confusion de los diferentes modelos segun la variable trained_model_dict

    :param x_eval: Contiene las variables de cada observacion
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    :param number_classes: Numero de clases que contiene los datos 
    """

    labels = ["CN","MCI","AD"]
    if number_classes == 2 :
        labels = ["sMCI", "pMCI"]

    for name, model in trained_model_dict.items():
        plot.plot_confusion_matrix(y_eval, model.predict(x_eval), DEST_FOLDER, name + using_DL, labels)


def show_result_bar_graph(x_eval, y_eval, trained_model_dict, using_DL, number_classes):
    """
    Generacion de graficas de barras de los diferentes modelos segun la variable trained_model_dict

    :param x_eval: Contiene las variables de cada observacion
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    :param number_classes: Numero de clases que contiene los datos 
    """

    # Obtenemos las metricas, los valores predecidos y el modelo ya entrenado
    accuracy, precision, recall, f1_score, x_ticks = [], [], [], [], []
    for name, model in trained_model_dict.items():
        metrics = get_metrics_model(model, x_eval, y_eval, name + using_DL)
        accuracy.append(metrics[0])
        precision.append(metrics[1])
        recall.append(metrics[2])
        f1_score.append(metrics[3])

        x_ticks.append(name + using_DL)
    
    
    # Ploteamos el grafico de barras
    plot.plot_bar_graph(5, 4, [accuracy, precision, recall, f1_score], ['r', 'g', 'b', 'y'],
        ["Accuracy", "Precision", "Recall", "F1Score"], x_ticks, DEST_FOLDER, using_DL)


def show_result_roc_curves(x_eval, y_eval, trained_model_dict, using_DL, number_classes):
    """
    Generacion de curvas ROC de los diferentes modelos segun la variable trained_model_dict

    :param x_eval: Contiene las variables de cada observacion
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    :param number_classes: Numero de clases que contiene los datos 
    """

    # Obtenemos true positive rate, false positive rate y roc auc de los diferentes modelos
    fpr_list, tpr_list, roc_auc_list, model_list = [], [], [], []
    for name, model in trained_model_dict.items():
        fpr, tpr, roc_auc = got_roc_values_multiclass(y_eval, model.predict_proba(x_eval), number_classes)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        model_list.append(name + using_DL + ' (area = %0.3f)')
    # Ploteamos curvas ROC
    plot.plot_roc_curves(5,fpr_list, tpr_list, roc_auc_list, model_list, DEST_FOLDER, using_DL)



def show_cohen_kappa(x_eval, y_eval, trained_model_dict, using_DL):
    """
    Generacion de valores cohen Kappa de los diferentes modelos segun la variable trained_model_dict

    :param x_eval: Contiene las variables de cada observacion
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param trained_model_dict: Modelos de aprendizaje tradicional entrenados
    :param using_DL: String, indica si utilizamos el Autoencoder o no
    """
    for name, model in trained_model_dict.items():
        print("Cohen Kappa Score obtained by " + name + using_DL + ": " + str(cohen_kappa_score(model.predict(x_eval), y_eval)))
        

def compare_model_bar_graph(x_eval, reduced_eval, y_eval, 
        autoencoder, model, model_reduce,
        model_name):
    """
    Comparacion de modelos mediante Gráficos de barras, se obtiene las metricas del modelo entrenado
    y se genera la gráfica de barra

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param model: Modelo de aprendizaje tradicional entrenado
    :param model_reduce:  Modelo de aprendizaje tradicional entrenado con un conjunto de datos de dimensionalidad reducida
    :param model_name: Nombre del modelo
    """

    model_baseline_metrics = get_metrics(y_eval, model.predict(x_eval))
    model_reduce_metrics = get_metrics(y_eval, model_reduce.predict(reduced_eval))
    autoencoder_metrics = get_metrics(autoencoder.predict_fine_tuning(x_eval, 0), y_eval)

    # Dividimos las metricas para cada uno
    accuracy = [model_baseline_metrics[0], autoencoder_metrics[0], model_reduce_metrics[0]]
    precision = [model_baseline_metrics[1], autoencoder_metrics[1], model_reduce_metrics[1]]
    recall = [model_baseline_metrics[2], autoencoder_metrics[2], model_reduce_metrics[2]]
    f1_score = [model_baseline_metrics[3], autoencoder_metrics[3], model_reduce_metrics[3]]
    x_ticks = [model_name,'Autoencoder', model_name + ' + AE']

    plot.plot_bar_graph(3, 4, [accuracy, precision, recall, f1_score], ['r', 'g', 'b', 'y'],
        ["Accuracy", "Precision", "Recall", "F1Score"], x_ticks, DEST_FOLDER, "compare" + model_name)



def compare_model_roc_curve(x_eval, reduced_eval, y_eval, 
                            autoencoder, model, model_reduce,
                            model_name, number_classes):
    """
    Comparacion de modelos mediante curvas ROC, se obtiene las metricas del modelo entrenado
    y se genera curva ROC

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param model: Modelo de aprendizaje tradicional entrenado
    :param model_reduce:  Modelo de aprendizaje tradicional entrenado con un conjunto de datos de dimensionalidad reducida
    :param model_name: Nombre del modelo
    :param number_classes: Numero de clases que contiene los datos 
    """

    bsln_fpr, bsln_tpr, bsln_roc_auc = got_roc_values_multiclass(y_eval, model.predict_proba(x_eval), number_classes)
    ae_fpr, ae_tpr, ae_roc_auc = got_roc_values_multiclass(y_eval, autoencoder.predict_proba(x_eval,0), number_classes)
    red_fpr, red_tpr, red_roc_auc = got_roc_values_multiclass(y_eval, model_reduce.predict_proba(reduced_eval) , number_classes)

    
    fpr_list = [bsln_fpr, ae_fpr, red_fpr]
    tpr_list = [bsln_tpr, ae_tpr, red_tpr]
    roc_auc_ist = [bsln_roc_auc, ae_roc_auc, red_roc_auc]
    model_list = ['Baseline ' + model_name + ' Mean (area = %0.3f)', 'Autoencoder (area = %0.3f)', model_name + ' + AE (area = %0.3f)']

    plot.plot_roc_curves(3,fpr_list, tpr_list, roc_auc_ist, model_list, DEST_FOLDER, "compare" + model_name)

def compare_model_mauc(x_eval, reduced_eval, y_eval, 
        autoencoder, model, model_reduce,
        model_name, number_classes):
    """
    Comparacion de modelos mediante obtencion del valor MAUC, generamos grafica de barras segun el valor MAUC del modelo

    :param x_eval: Contiene las variables de cada observacion
    :param reduced_eval: Contiene las variables de cada observacion con una dimensionalidad reducida
    :param y_eval: Contiene la clase a la que pertenece cada observacion
    :param autoencoder: Modelo autoencoder entrenado
    :param model: Modelo de aprendizaje tradicional entrenado
    :param model_reduce:  Modelo de aprendizaje tradicional entrenado con un conjunto de datos de dimensionalidad reducida
    :param model_name: Nombre del modelo
    :param number_classes: Numero de clases que contiene los datos 
    """

    bsl_MAUC = MAUC(got_zip_class_prob(y_eval,  model.predict_proba(x_eval)), number_classes)
    ae_MAUC = MAUC(got_zip_class_prob(y_eval, autoencoder.predict_proba(x_eval,0)), number_classes)
    red_MAUC = MAUC(got_zip_class_prob(y_eval, model_reduce.predict_proba(reduced_eval)) , number_classes)

    mauc_list = [bsl_MAUC, ae_MAUC, red_MAUC]
    x_ticks = [model_name,'Autoencoder', model_name + ' + AE']

    plot.plot_bar_graph(3, 2, [[], mauc_list] , ['w', 'b'], ["", 'MAUC'], x_ticks, DEST_FOLDER, "compareMAUC" + model_name)

def evaluate_AE_with_shallow(x_train, x_eval, y_train, y_eval,
        learning_rate, batch_size, 
        epochs, dropout_value,
        size_layers_used, latent_space,
        activation_func, number_classes, verbose):
    """
    Mostramos por pantalla las metricas obtenidas a partir de un modelo si utilizamos
    el proceso K fold cross validation

    :param x_train e x_eval: Contiene las variables de cada observacion
    :param y_train e y_eval : Contiene la clase a la que pertenece cada observacion
    :param learning_rate: Ratio de aprendizaje del autoencoder
    :param batch_size: Numero de batch size para el entrenamiento
    :param epochs: Numero de epocas del entrenamiento
    :param dropout_value: Valor dropout para las capas del autoencoder
    :param size_layers_used: Array que contiene el numero de cada una de las capas
    :para latent_space: Numero de neuronas de la capa bottleneck
    :param activation_func: Funcion de activacion de las capas
    :param number_classes: Numero de clases que contiene el problema
    :param verbose: False si no quieres que se muestren cosas por pantallas, True en caso contrario
    """

    autoencoder = train_autoencoder(x_train, y_train, 
        learning_rate, batch_size, epochs, dropout_value, 
        size_layers_used, latent_space, activation_func, number_classes, 
        True, verbose, 0)

    labels = ["CN","MCI","AD"]
    if number_classes == 2 :
        labels = ["sMCI", "pMCI"]
    plot.plot_confusion_matrix(y_eval, autoencoder.predict_fine_tuning(x_eval, 0), DEST_FOLDER, "AE", labels)
    print("AUTOENCODER MODEL METRICS: ")
    show_metrics(get_metrics(autoencoder.predict_fine_tuning(x_eval, 0), y_eval))
    print("Cohen Kappa score: " + str(cohen_kappa_score(autoencoder.predict_fine_tuning(x_eval, 0), y_eval)))


    # Devolvemos los datos reducidos mediante el autoencoder
    reduced_train = autoencoder.return_reduce_attribute(x_train)
    reduced_eval = autoencoder.return_reduce_attribute(x_eval)

    trained_model_dict = { "KNN": KNeighborsClassifier(n_neighbors = number_classes).fit(x_train, y_train),
        "SVM": svm.SVC(kernel='linear', probability = True).fit(x_train, y_train),
        "DT": DecisionTreeClassifier().fit(x_train, y_train),
        "RF": RandomForestClassifier().fit(x_train, y_train),
        "GB": GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3).fit(x_train, y_train)}

    trained_model_dict_reduced = { "KNN": KNeighborsClassifier(n_neighbors = number_classes).fit(reduced_train, y_train),
        "SVM": svm.SVC(kernel='linear', probability = True).fit(reduced_train, y_train),
        "DT": DecisionTreeClassifier().fit(reduced_train, y_train),
        "RF": RandomForestClassifier().fit(reduced_train, y_train),
        "GB": GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3).fit(reduced_train, y_train)}
    
    # Evaluacion Modelo a Modelo
    compare_models_roc_curve(x_eval, reduced_eval, y_eval, 
        autoencoder, trained_model_dict, trained_model_dict_reduced, 
        number_classes)
    compare_models_bar_graph(x_eval, reduced_eval, y_eval, 
        autoencoder, trained_model_dict, trained_model_dict_reduced)


    # Evaluacion modelo entero
    show_result(x_eval, y_eval, trained_model_dict, "", number_classes)
    show_result(reduced_eval, y_eval, trained_model_dict_reduced, " + AE", number_classes)

    show_cohen_kappa(x_eval, y_eval, trained_model_dict, "")
    show_cohen_kappa(reduced_eval, y_eval, trained_model_dict_reduced, " + AE")

    # Evaluacion de tiempo de modelos
    show_time(x_train, y_train, trained_model_dict, "")
    show_time(reduced_train, y_train, trained_model_dict_reduced, " + AE")

def usageCommand():
    """
    Muestra por pantalla como se debe utilizar los comandos para generar datos o evaluar
    los modelos
    """

    usage = "python3 main.py -GENERATE [-C] [-MRI] [-PET] [-DTI] [-BIO] [-ALL] [-sMCIpMCI] [-DELETE] [-useDX]"
    print("Usage for generate Data: " + usage, file=sys.stderr)
    usage = "python3 main.py -LOAD pathTrainData pathTestData [-KFOLD] [KFOLDAE] [-COMPARE] [-BAYESIAN] [-PAROPTI] [-sMCIpMCI]"
    print("Usage for train Data: " + usage, file=sys.stderr)
    exit()

def main(argv):
    """
    Funcion main del programa
    Esta funcion se encarga de segun los argumentos que se dan por linea de comandos,
    el programa hara varias funciones:
    - Generar un conjunto de datos procesados para el autoencoder a partir de los datos
    TADPOLE CHALLENGE, dependiendo de que flags se utilicen se tendran en cuenta unos
    atributos u otros o tambien el tipo de problema a evaluar
    - Realizar una comparacion entre varios modelos, ya sea autoencoder y red neuronal
    O autoencoder con varios modelos o una optimizacion de parametros del autoencoder o
    una ejecucion kfold de un modelo Random Forest 

    :param argv: Array que contiene los argumentos que se dan por linea de comandos
    """
    global DEST_FOLDER

    # Error por pantalla al usar mal el comando de ejecucion del programa
    if (len(argv) > 8 or len(argv) == 0):
        usageCommand()
    dataset = Datasets()

    # Etapa de generacion de un conjunto de datos validos para el autoencoder
    if argv[0] == "-GENERATE":
        ad_or_mci = 0
        using_DX = 0

        # Paths del origen de los datos
        clinic_paths= ["../Data/TADPOLE_D1_D2.csv", "../Data/TADPOLE_D3.csv", "../Data/TADPOLE_D4_corr.csv"]
        features =  read_file("../features/others")

        # Atributos que se tienen en cuenta para el procesado de datos
        for i in range(1, len(argv)):
            if (argv[i] == "-C"):
                features = features + read_file("../features/cognitive")

            elif (argv[i] == "-MRI"):
                features = features + read_file("../features/UCSFFSL")
                features = features + read_file("../features/UCSFFSX")
            
            elif (argv[i] == "-PET"):
                features = features + read_file("../features/BAIPETNMRC")
                features = features + read_file("../features/UCBERKELEYAV45-1451")
                
            elif (argv[i] == "-DTI"):
                features = features + read_file("../features/DTIROI")

            elif (argv[i] == "-BIO"):
                features = features + read_file("../features/Biomarkers")

            elif (argv[i] == "-sMCIpMCI"):
                ad_or_mci = 1
            
            elif (argv[i] == "-useDX"):
                using_DX = 1
            
            elif (argv[i] == "-ALL"):
                features = features + read_file("../features/cognitive")
                features = features + read_file("../features/UCSFFSL")
                features = features + read_file("../features/UCSFFSX")
                features = features + read_file("../features/BAIPETNMRC")
                features = features + read_file("../features/UCBERKELEYAV45-1451")
                features = features + read_file("../features/DTIROI")
                features = features + read_file("../features/Biomarkers")

            # Flag para eliminar los datos procesados (si se quiere volver a ejecutar)
            elif (argv[i] == "-DELETE"):
                train_data_Path = "../TrainTadpole.csv"
                eval_data_Path = "../EvalTadpole.csv"
                if os.path.exists(train_data_Path):
                    os.remove(train_data_Path)
                if os.path.exists(eval_data_Path):
                    os.remove(eval_data_Path)

        # Creamos el dataset de los datos médicos y prepocesamos los datos
        [train_data, eval_data] = dataset.load_TADPOLE(clinic_paths, features, ad_or_mci, using_DX)
    #Etapa de evaluacion de modelos
    elif argv[0] == "-LOAD":

        # Obtenemos los conjuntos de datos y los seperamos
        train_data = pd.read_csv(argv[1], sep = ";")
        eval_data = pd.read_csv(argv[2], sep = ";")
        [x_train, y_train] = dataset.divide_data(train_data)
        [x_eval, y_eval] = dataset.divide_data(eval_data)

        # Datos para el autoencoder 
        layer_sizes = [600,300,200]
        latent_space = 150
        number_classes = 3

        # Flags
        optimization_used = 0 # 0 Grid Search 1 Bayesian Search
        execution_mode = 0
        model_Kfold = "autoencoder"
        for i in range(1, len(argv)):
            if (argv[i] == "-sMCIpMCI"):
                number_classes = 2
                DEST_FOLDER = "sMCI_pMCI"
            elif (argv[i] == "-KFOLDAE"):
                execution_mode = 1
                model_Kfold = "autoencoder"
            elif(argv[i] == "-KFOLD"):
                execution_mode = 1
            elif (argv[i] == "-COMPARE"):
                execution_mode = 2
            elif (argv[i] == "-PAROPTI"):
                execution_mode = 3
            elif (argv[i] == "-BAYESIAN"):
                optimization_used = 1
                
        # Dependiendo del flag hara una ejecucion u otra
        if execution_mode == 1:
            if(model_Kfold == "autoencoder"):
                kFold_cross_validation_DLmodel(x_train, y_train, 
                    LEARNINGRATE, BATCHSIZE, 
                    EPOCHS, DROPOUTVALUE,
                    layer_sizes, latent_space,
                    "relu", number_classes, False)
                hyper_parameters_autoencoder(x_train, y_train, layer_sizes, latent_space, "relu")
            else:
                # Evaluacion de metricas del modelo Random Forest mediante kfold
                kFold_cross_validation_shallow(SHALLOWMODELDICT[model_Kfold], x_train, y_train, model_Kfold)
        elif execution_mode == 2:
            # Evaluacion entre Autoencoder y Red Neuronal. Se obtienen metricas segun kfdol
            kFold_cross_validation_DLmodel(x_train, y_train,
                LEARNINGRATE, BATCHSIZE, 
                EPOCHS, DROPOUTVALUE,
                layer_sizes,  latent_space, 
                "relu", number_classes,  True)
            
            kFold_cross_validation_DLmodel(x_train, y_train,
                LEARNINGRATE, BATCHSIZE, 
                EPOCHS, DROPOUTVALUE,
                layer_sizes,  latent_space, 
                "relu", number_classes, False)
        elif execution_mode == 3:
            if(optimization_used):
                # Obtencion de hyper parametros del autoencoder mediante bayesian optimization
                hyper_parameters_optimization(x_train)
            else:
                # Obtencion de metricas del modelo Autoencoder
                hyper_parameters_autoencoder(x_train, y_train, layer_sizes, latent_space, "relu")
        else:
            # Ejecucion por defecto, comparamos Autoencoder con modelos de aprendizaje tradicionales
            # Evaluando ambos modelos y viendo si es posible utilizar el autoencoder como modelo de apoyo
            evaluate_AE_with_shallow(x_train, x_eval, y_train, y_eval,
                LEARNINGRATE, BATCHSIZE, 
                EPOCHS, DROPOUTVALUE,
                layer_sizes,  latent_space, 
                "relu", number_classes, True)
    # Error para mostrar por pantalla como se utiliza los comandos
    else:
        usageCommand()


if __name__ == "__main__":
    main(sys.argv[1:])
    