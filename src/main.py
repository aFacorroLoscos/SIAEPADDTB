# Bibliotecas generales
import os
import sys
import numpy as np
import math
import shap
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
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_curve, auc, mean_squared_error, cohen_kappa_score

# Hyper parameters optimization
from skopt import BayesSearchCV 
from scikeras.wrappers import KerasRegressor
from skopt.space import Real, Integer

from scipy import stats
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

"""
    Pre: fileNamePath es un path valido a un archivo de texto
    Post: Devuelve una lista de datos que contiene las palabras que se encontraban
          dentro del archivo de texto
"""
def read_file(file_name_path):
        file_obj = open(file_name_path, "r")
        words = file_obj.read().splitlines()
        file_obj.close()
        return words


def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])

"""
    Pre: x_train contiene las variables de cada observacion, y_train contiene la clase a la que
         pertenece cada observacion.
         learning_rate, batch_size, epochs y num_out_puts  debe ser 
         estrictamente mayor que 0. El tamaño de endcoder_func, decoder_func y 
         size_layers_used debe ser el mismo.  
    Post: Devuelve un autoencoder entrenado mediante fine tuning
"""
def train_autoencoder(x_train, y_train, learning_rate, 
                      batch_size, epochs, dropout,
                     size_layers_used, latent_space, activation_func,
                     number_classes, autoencoder_model, 
                     verbose, verbose_train):
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


"""
    Pre: x_train contiene las variables de cada observacion, y_train contiene la clase a la que
         pertenece cada observacion.
    Post: Muestra por pantala la distribucion de valores de train y test y la exactitud
          de cada uno de los folds y la exactitud del algoritmo Cross Validation
"""
def kFold_cross_validation_DLmodel(x_train, y_train, lr, 
                                    batch_size, epochs, dropout_value,
                                    size_layers_used, latent_space,
                                    activation_func, number_classes, autoencoder_model):
    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(x_train, y_train)
    accuracys = []
    precisions = []
    recalls = []
    F1scores = []  
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
    model_name = "Autoencoder" if autoencoder_model else "Neural Network"

    print("Result obtained using " + model_name + " model:")
    print('Cross-Validation accuracy: %.3f +/- %.3f' %(np.mean(accuracys), np.std(accuracys)))
    print('Cross-Validation precision: %.3f +/- %.3f' %(np.mean(precisions), np.std(precisions)))
    print('Cross-Validation recall: %.3f +/- %.3f' %(np.mean(recalls), np.std(recalls)))
    print('Cross-Validation meanF1: %.3f +/- %.3f\n ' %(np.mean(F1scores), np.std(F1scores)))


def kFold_cross_validation_shallow(model, x_values, y_values, model_name):
    
    strtfdKFold = StratifiedKFold(n_splits=10)
    
    kfold = strtfdKFold.split(x_values, y_values)

    accuracys = []
    precisions = []
    recalls = []
    meansF1 = [] 
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
    
    print("Result obtained using " + model_name + " model:")
    print('Cross-Validation accuracy: %.3f +/- %.3f' %(np.mean(accuracys), np.std(accuracys)))
    print('Cross-Validation precision: %.3f +/- %.3f' %(np.mean(precisions), np.std(precisions)))
    print('Cross-Validation recall: %.3f +/- %.3f' %(np.mean(recalls), np.std(recalls)))
    print('Cross-Validation meanF1: %.3f +/- %.3f\n ' %(np.mean(meansF1), np.std(meansF1)))

def hyper_parameters_autoencoder(x_train, y_train, size_layers, latent_space, encoder_act_func, decoder_act_func):
    learningrate = [0.01, 0.001, 0.0001]
    learningrate_fine_tuning = [0.001, 0.0001, 0.00001]
    batch_size = [32]
    dropout_values = [0.3,0.4]
    epochs = [15,20,25]

    best_precision = 0
    best_hyper_parameters = [0, 0, 0, 0, 0]
    for j in range(len(batch_size)):
        for l in range(len(epochs)):
            for i in range(len(learningrate)):
                for k in range(len(dropout_values)):
                    print("Using: Dropout " + str(dropout_values[k]) + " , LRate: " + str(learningrate[i]) + ", Epochs: " + str(epochs[l]) + " , BatchSize: " + str(batch_size[j]), flush=True)
                    current_precision = kFold_cross_validation_autoencoder(x_train, y_train, 
                                                                          learningrate[i], learningrate_fine_tuning[i], 
                                                                          batch_size[j], epochs[l], dropout_values[k],
                                                                          size_layers, latent_space,
                                                                          encoder_act_func, decoder_act_func, false)
                    if current_precision > best_precision:
                        best_hyper_parameters[0] = learningrate[i]
                        best_hyper_parameters[1] = learningrate_fine_tuning[i]
                        best_hyper_parameters[2] = batch_size[j]
                        best_hyper_parameters[3] = dropout_values[k]
                        best_hyper_parameters[4] = epochs[l]
                        bestPrecision = current_precision

    print("Best F1 Score: " + str(best_precision))
    print("Best Hyperparameters: ")
    print("Learning Rate: " + str(best_hyper_parameters[0]))
    print("Learning Rate FT: " + str(best_hyper_parameters[1]))
    print("Batch Size: " + str(best_hyper_parameters[2]))
    print("Dropout: " + str(best_hyper_parameters[3]))
    print("Epochs: " + str(best_hyper_parameters[4]))

def create_autoencoder_hyper_par(dropout_value):
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


    callback =  keras.callbacks.EarlyStopping(monitor = "loss", mode = "min", verbose = 1, patience = 10, min_delta = 0.01)

    kfold = KFold(n_splits = 10, random_state = 42, shuffle = True)

    search_spaces = {
        "optimizer__learning_rate" : (1e-3, 0.1, "log-uniform") ,
        "batch_size" : [16,32,64],
        "model__dropout_value" : [0.2,0.3,0.4]
    }


    autoencoder_model = KerasRegressor(create_autoencoder_hyper_par, loss = "mean_absolute_error", optimizer = "adam", epochs = 50, callbacks = [callback], verbose = 0, random_state = 42)
    
    opt = BayesSearchCV(
        estimator = autoencoder_model,
        search_spaces = search_spaces,
        n_jobs = -1,
        cv = kfold,
        verbose = 0
    )

    opt.fit(x_train, x_train)

    print("Best val. score: %s" % opt.best_score_)
    print("Best params obtained: %s" % str(opt.best_params_))



"""
    Pre: y_test es un conjunto de variables de respuesta y y_pred es un conjunto de variables predictoras
    Post: Devuelve un vector que contiene las metricas comparando las variables predictoras y respuestas
"""
def get_metrics(y_test, y_pred):
    clasifier_metrics = []
    clasifier_metrics.append(accuracy_score(y_test, y_pred))
    clasifier_metrics.append(precision_score(y_test, y_pred, average='weighted', zero_division = 0))
    clasifier_metrics.append(recall_score(y_test, y_pred, average='weighted', zero_division = 0))
    clasifier_metrics.append(f1_score(y_test, y_pred, average='weighted'))
    return clasifier_metrics

def show_metrics(metrics):
    print("Accuracy: " + str(round(metrics[0],3)))
    print("Precision: " + str(round(metrics[1],3)))
    print("Recall: " + str(round(metrics[2],3)))
    print("F1 Score: " + str(round(metrics[3],3)))
"""
    Pre: x_train e x_test contiene las variables de cada observacion, y_train e y_test contiene la clase a la que
         pertenece cada observacion.
    Post: Devuelve las metricas del modelo de clasificación
"""
def get_metrics_model(model, x_test, y_test, model_name):
    metrics = get_metrics(y_test, model.predict(x_test))
    print("Metrics obtained " + model_name + ":")
    show_metrics(metrics)
    return metrics

def train_show_time(model, x_train, y_train, model_name):
    t0  = time.time()
    model.fit(x_train, y_train)
    print("Training Time from " + model_name + ":", time.time()-t0)

def got_roc_values(x_data, y_data):
    fpr, tpr, thresholds = roc_curve(x_data, y_data)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def got_roc_values_multiclass(y_test, y_scores, num_classes):
    y_test_onehot = one_hot(y_test, num_classes)

    fpr, tpr, _ = roc_curve(y_test_onehot.ravel(), y_scores.ravel())

    return fpr, tpr, auc(fpr, tpr)

def got_zip_class_prob(class_list, prob_ist):
    zip_true_label_probs = []
    for x in range(len(class_list)):
        zip_true_label_probs += [(class_list[x], list(prob_ist[x]))]
    
    return zip_true_label_probs


def compare_models_bar_graph(x_eval, reduced_eval, y_eval, 
                             autoencoder, trained_model_dict, trained_model_dict_reduced):
    
    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_bar_graph(x_eval, reduced_eval, y_eval, 
                                autoencoder, model, model_reduce,
                                name_1)

def compare_models_roc_curve(x_eval, reduced_eval, y_eval, 
                             autoencoder, trained_model_dict, trained_model_dict_reduced, 
                             number_classes):

    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_roc_curve( x_eval, reduced_eval, y_eval, 
                                autoencoder, model, model_reduce,
                                name_1,number_classes)

def compare_models_MAUC(x_eval, reduced_eval, y_eval, 
                        autoencoder, trained_model_dict, trained_model_dict_reduced, 
                        number_classes):
    
    for (name_1, model), (name_2, model_reduce) in zip(trained_model_dict.items(), trained_model_dict_reduced.items()):
        compare_model_mauc(x_eval, reduced_eval, y_eval, 
                           autoencoder, model, model_reduce,
                           name_1,number_classes)
    

def show_time(x_train, y_train, trained_model_dict, using_DL):
    for name, model in trained_model_dict.items():
        train_show_time(model, x_train, y_train, name + using_DL)

def show_result(x_eval, y_eval, trained_model_dict, using_DL, number_classes):

    plot_matrix(x_eval, y_eval, trained_model_dict, using_DL, number_classes)
    
    show_result_bar_graph(x_eval, y_eval, trained_model_dict, using_DL, number_classes)
             
    show_result_roc_curves(x_eval, y_eval, trained_model_dict, using_DL, number_classes)

def plot_matrix(x_test, y_test, trained_model_dict, using_DL, number_classes):
    labels = ["CN","MCI","AD"]
    if number_classes == 2 :
        labels = ["sMCI", "pMCI"]

    for name, model in trained_model_dict.items():
        plot.plot_confusion_matrix(y_test, model.predict(x_test), DEST_FOLDER, name + using_DL, labels)


def show_result_bar_graph(x_eval, y_eval, trained_model_dict, using_DL, number_classes):
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
    

    fpr_list, tpr_list, roc_auc_list, model_list = [], [], [], []
    for name, model in trained_model_dict.items():
        fpr, tpr, roc_auc = got_roc_values_multiclass(y_eval, model.predict_proba(x_eval), number_classes)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)

        model_list.append(name + using_DL + ' (area = %0.3f)')
    
    plot.plot_roc_curves(5,fpr_list, tpr_list, roc_auc_list, model_list, DEST_FOLDER, using_DL)



def show_cohen_kappa(x_eval, y_eval, trained_model_dict, using_DL):

    for name, model in trained_model_dict.items():
        print("Cohen Kappa Score obtained by " + name + using_DL + ": " + str(cohen_kappa_score(model.predict(x_eval), y_eval)))
        

def compare_model_bar_graph(x_eval, reduced_eval, y_eval, 
                            autoencoder, model, model_reduce,
                            model_name):
    
    model_baseline_metrics = get_metrics(y_eval, model.predict(x_eval))

    model_reduce_metrics = get_metrics(y_eval, model_reduce.predict(reduced_eval))
    
    autoencoder_metrics = get_metrics(autoencoder.predict_fine_tuning(x_eval, 0), y_eval)

    # Dividimos las metricas para cada uno
    accuracy = [model_baseline_metrics[0], autoencoder_metrics[0], model_reduce_metrics[0]]
    precision = [model_baseline_metrics[1], autoencoder_metrics[1], model_reduce_metrics[1]]
    recall = [model_baseline_metrics[2], autoencoder_metrics[2], model_reduce_metrics[2]]
    f1_score = [model_baseline_metrics[3], autoencoder_metrics[3], model_reduce_metrics[3]]
    x_ticks = [model_name,'Autoencoder', model_name + ' + DL']

    plot.plot_bar_graph(3, 4, [accuracy, precision, recall, f1_score], ['r', 'g', 'b', 'y'],
                 ["Accuracy", "Precision", "Recall", "F1Score"], x_ticks, DEST_FOLDER, "compare" + model_name)



def compare_model_roc_curve(x_eval, reduced_eval, y_eval, 
                            autoencoder, model, model_reduce,
                            model_name, number_classes):

    bsln_fpr, bsln_tpr, bsln_roc_auc = got_roc_values_multiclass(y_eval, model.predict_proba(x_eval), number_classes)

    ae_fpr, ae_tpr, ae_roc_auc = got_roc_values_multiclass(y_eval, autoencoder.predict_proba(x_eval,0), number_classes)

    red_fpr, red_tpr, red_roc_auc = got_roc_values_multiclass(y_eval, model_reduce.predict_proba(reduced_eval) , number_classes)

    
    fpr_list = [bsln_fpr, ae_fpr, red_fpr]
    tpr_list = [bsln_tpr, ae_tpr, red_tpr]
    roc_auc_ist = [bsln_roc_auc, ae_roc_auc, red_roc_auc]
    model_list = ['Baseline ' + model_name + ' Mean (area = %0.3f)', 'Autoencoder (area = %0.3f)', model_name + ' + DL (area = %0.3f)']

    plot.plot_roc_curves(3,fpr_list, tpr_list, roc_auc_ist, model_list, DEST_FOLDER, "compare" + model_name)

def compare_model_mauc(x_eval, reduced_eval, y_eval, 
                     autoencoder, model, model_reduce,
                     model_name, number_classes):

    bsl_MAUC = MAUC(got_zip_class_prob(y_eval,  model.predict_proba(x_eval)), number_classes)

    ae_MAUC = MAUC(got_zip_class_prob(y_eval, autoencoder.predict_proba(x_eval,0)), number_classes)

    red_MAUC = MAUC(got_zip_class_prob(y_eval, model_reduce.predict_proba(reduced_eval)) , number_classes)

    mauc_list = [bsl_MAUC, ae_MAUC, red_MAUC]
    x_ticks = [model_name,'Autoencoder', model_name + ' + DL']

    plot.plot_bar_graph(3, 2, [[], mauc_list] , ['w', 'b'], ["", 'MAUC'], x_ticks, DEST_FOLDER, "compareMAUC" + model_name)

def evaluate_AE_with_shallow(x_train, x_eval, y_train, y_eval,
                             learning_rate, batch_size, 
                             epochs, dropout_value,
                             size_layers_used, latent_space,
                             activation_func, number_classes, verbose):

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
                           "GB": GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3).fit(x_train, y_train)
    }

    trained_model_dict_reduced = { "KNN": KNeighborsClassifier(n_neighbors = number_classes).fit(reduced_train, y_train),
                                   "SVM": svm.SVC(kernel='linear', probability = True).fit(reduced_train, y_train),
                                   "DT": DecisionTreeClassifier().fit(reduced_train, y_train),
                                   "RF": RandomForestClassifier().fit(reduced_train, y_train),
                                   "GB": GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3).fit(reduced_train, y_train)
    }
    
    # Evaluacion Modelo a Modelo
    compare_models_roc_curve(x_eval, reduced_eval, y_eval, 
                             autoencoder, trained_model_dict, trained_model_dict_reduced, 
                             number_classes)
    compare_models_bar_graph(x_eval, reduced_eval, y_eval, 
                             autoencoder, trained_model_dict, trained_model_dict_reduced)


    # Evaluacion modelo entero
    show_result(x_eval, y_eval, trained_model_dict, "", number_classes)
    show_result(reduced_eval, y_eval, trained_model_dict_reduced, " + DL", number_classes)

    show_cohen_kappa(x_eval, y_eval, trained_model_dict, "")
    show_cohen_kappa(reduced_eval, y_eval, trained_model_dict_reduced, " + DL")

    # Evaluacion de tiempo de modelos
    show_time(x_train, y_train, trained_model_dict, "")
    show_time(reduced_train, y_train, trained_model_dict_reduced, " + DL")

def usageCommand():
    usage = "py main.py -GENERATE [-C] [-MRI] [-PET] [-DTI] [-BIO] [-sMCIpMCI] [-DELETE] [-useDX]]"

    print("Usage for generate Data: " + usage, file=sys.stderr)

    usage = "py main.py -LOAD pathTrainData pathTestData [-COMPARE] [-KFOLD] [-sMCIpMCI]"

    print("Usage for train Data: " + usage, file=sys.stderr)
    exit()

def main(argv):
    global DEST_FOLDER

        # Error por pantalla al usar mal el comando de ejecucion del programa
    if (len(argv) > 8 or len(argv) == 0):
        usageCommand()
    dataSet = Datasets()
    # Si los argumentos son correctos, se anyadiran los features a tener en cuenta
    if argv[0] == "-GENERATE":
        ad_or_mci = 0
        using_DX = 0

        clinicPaths= ["../Data/TADPOLE_D1_D2.csv", "../Data/TADPOLE_D3.csv", "../Data/TADPOLE_D4_corr.csv"]
        features =  read_file("../features/others")
        feature_type_path = "../Feature_Type.csv"

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

            elif (argv[i] == "-DELETE"):
                train_data_Path = "../TrainTadpole.csv"
                eval_data_Path = "../EvalTadpole.csv"
                if os.path.exists(train_data_Path):
                    os.remove(train_data_Path)
                if os.path.exists(eval_data_Path):
                    os.remove(eval_data_Path)

        # Creamos el dataset de los datos médicos y prepocesamos los datos
        [trainData, evalData] = dataSet.loadTADPOLE(clinicPaths, features, feature_type_path, ad_or_mci, using_DX)
    elif argv[0] == "-LOAD":

        trainData = pd.read_csv(argv[1], sep = ";")
        evalData = pd.read_csv(argv[2], sep = ";")


        print(trainData["Diagnosis"].value_counts())
        print(evalData["Diagnosis"].value_counts())

        [xTrain, yTrain] = dataSet.divideData(trainData)
        [xEval, yEval] = dataSet.divideData(evalData)



        layer_sizes = [600,300,200]
        latent_space = 150

        number_classes = 3
        optimization_used = 0 # 0 Grid Search 1 Bayesian Search
        execution_mode = 0
        model_Kfold = "autoencoder"

        for i in range(1, len(argv)):
            if (argv[i] == "-sMCIpMCI"):
                number_classes = 2
                DEST_FOLDER = "sMCI_pMCI"
            elif (argv[i] == "-KFOLDAE"):
                execution_mode = 1
            elif(argv[i] == "-KFOLDSHALLOW"):
                execution_mode = 1
                model_Kfold = argv[i + 1]
            elif (argv[i] == "-COMPARE"):
                execution_mode = 2
            elif (argv[i] == "-PAROPTI"):
                execution_mode = 3
            elif (argv[i] == "-BAYESIAN"):
                optimization_used = 1
                
        

        if execution_mode == 1:
            if(model_Kfold == "autoencoder"):
                hyper_parameters_autoencoder(xTrain, yTrain, layer_sizes, latent_space, "relu", "relu")
            else:
                kFold_cross_validation_shallow(SHALLOWMODELDICT[model_Kfold], xTrain, yTrain, model_Kfold)
        elif execution_mode == 2:
            kFold_cross_validation_DLmodel(xTrain, yTrain,
                           LEARNINGRATE, BATCHSIZE, 
                           EPOCHS, DROPOUTVALUE,
                           layer_sizes,  latent_space, 
                           "relu", number_classes,  True)
            
            kFold_cross_validation_DLmodel(xTrain, yTrain,
                           LEARNINGRATE, BATCHSIZE, 
                           EPOCHS, DROPOUTVALUE,
                           layer_sizes,  latent_space, 
                           "relu", number_classes, False)
        elif execution_mode == 3:
            if(optimization_used):
                hyper_parameters_optimization(xTrain)
            else:
                hyper_parameters_autoencoder(xTrain, yTrain, layer_sizes, latent_space, "relu", "relu")
        else:
            evaluate_AE_with_shallow(xTrain, xEval, yTrain, yEval,
                           LEARNINGRATE, BATCHSIZE, 
                           EPOCHS, DROPOUTVALUE,
                           layer_sizes,  latent_space, 
                           "relu", number_classes, True)

    else:
        usageCommand()


if __name__ == "__main__":
    main(sys.argv[1:])
    