# Bibliotecas generales
import os
import sys
import numpy as np
import math
import shap
import keras

# Biblioteca para Kfold, split de datos y labelice de las clases
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# Clases propias
from Autoencoders.aeClass import Autoencoder
from LoadData.loadMedicData import Datasets
from MAUC import MAUC 
import plot

# Shallow models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.base import clone

# Metrics
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_curve, auc

# Graph plotting
import matplotlib.pyplot as plt


# Variables GLOBALES PARA ENTRENAR EL AUTOENCODER
LEARNINGRATE = 0.0004
LEARNINGRATEFT = 0.00004
BATCHSIZE = 32
EPOCHS = 25
DROPOUTVALUE = 0.1

# Layers que se utilizan para cbig_cross_sectional_features

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
         learning_rate, learning_rate_ft, batch_size, epochs y num_out_puts  debe ser 
         estrictamente mayor que 0. El tamaño de endcoder_func, decoder_func y 
         size_layers_used debe ser el mismo.  
    Post: Devuelve un autoencoder entrenado mediante fine tuning
"""
def train_autoencoder(x_train, y_train, 
                     learning_rate, learning_rate_ft, 
                     batch_size, epochs, dropout,
                     size_layers_used, latent_space,
                     encoder_func, decoder_func,
                     verbose, verbose_mode, number_classes):
    # Creamos el autoencoder
    autoencoder = Autoencoder(
        input_shape = (np.shape(x_train)[1]),
        num_hidden_layers = len(size_layers_used),
        size_layers = size_layers_used,
        latent_space = latent_space,
        drop_out_value = dropout,
        encoder_activations_func = encoder_func,
        decoder_activations_func = decoder_func,
    )
    
    if verbose:
        autoencoder.summary()


    # Entrenamos el autoencoder
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs, verbose_mode)

    if verbose:
        autoencoder.obtain_history(0, 0, "lossHistory")

    # Entrenamos el autoencoder para que pueda
    # clasificar mediante fine tuning
    autoencoder.fine_tuning(number_classes)

    if verbose:
        autoencoder.summary_fine_tuning()

    autoencoder.compile_fine_tuning(learning_rate_ft)
    autoencoder.set_trainable(True)
    autoencoder.train_fine_tuning(x_train, y_train, batch_size, epochs, verbose_mode)

    if verbose:
        autoencoder.obtain_history(1, 0, "lossHistoryFT")
        autoencoder.obtain_history(1, 1, "lossAccuracyFT")

    return autoencoder



"""
    Pre: x_train contiene las variables de cada observacion, y_train contiene la clase a la que
         pertenece cada observacion.
    Post: Muestra por pantala la distribucion de valores de train y test y la exactitud
          de cada uno de los folds y la exactitud del algoritmo Cross Validation
"""
def kFold_cross_validation_autoencoder(x_train, y_train, lr, lr_fine_tuning, 
                                    batch_size, epochs, dropout_value,
                                    size_layers_used, latent_space,
                                    encoder_func, decoder_func):
    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(x_train, y_train)
    scores = [] 
    for k, (train, test) in enumerate(kfold):
        autoencoder = train_autoencoder(x_train[train], y_train[train], lr, lr_fine_tuning, 
                                       batch_size, epochs, dropout_value, 
                                       size_layers_used, latent_space,
                                       encoder_func, decoder_func, False, 0)
        
        y_pred = autoencoder.predict_fine_tuning(x_train[test], 0)

        score = autoencoder.get_f1_score(y_pred, y_train[test])
        scores.append(score)

        print('Fold: %2d, Training/Test Split Distribution: %s - %s, F1Score: %.3f' % (k+1, 
                                                                                        np.shape(y_train[train])[0],
                                                                                        np.shape(y_train[test])[0],
                                                                                        score), flush=True)
        
    print('\n\nCross-Validation F1Score: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)), flush=True)
    return(np.mean(scores))


def kFold_cross_validation_shallow(model, x_values, y_values, model_name):
    
    strtfdKFold = StratifiedKFold(n_splits=10)
    
    kfold = strtfdKFold.split(x_values, y_values)

    accuracys = []
    precisions = []
    recalls = []
    meansF1 = [] 
    for k, (train, test) in enumerate(kfold):

        metrics = get_metrics_model(model,
                                   x_values[train], y_values[train],
                                   x_values[test], y_values[test])
        
        accuracys.append(metrics[0])
        precisions.append(metrics[1])
        recalls.append(metrics[2])
        meansF1.append(metrics[3])
        
        # print('Fold: %2d, Training/Test Split Distribution: %s - %s, Accuracy: %.3f' % (k+1, 
        #                                                                                 np.shape(yTrain[train])[0],
        #                                                                                 np.shape(yTrain[test])[0],
        #                                                                                 metrics[0]))
    
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
                                                                          encoder_act_func, decoder_act_func)
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


"""
    Pre: y_test es un conjunto de variables de respuesta y y_pred es un conjunto de variables predictoras
    Post: Devuelve un vector que contiene las metricas comparando las variables predictoras y respuestas
"""
def get_metrics (y_test, y_pred):
    clasifier_metrics = []
    clasifier_metrics.append(accuracy_score(y_test, y_pred))
    clasifier_metrics.append(precision_score(y_test, y_pred, average='macro', zero_division = 0))
    clasifier_metrics.append(recall_score(y_test, y_pred, average='macro', zero_division = 0))
    clasifier_metrics.append(f1_score(y_test, y_pred, average='macro'))
    return clasifier_metrics
"""
    Pre: x_train e x_test contiene las variables de cada observacion, y_train e y_test contiene la clase a la que
         pertenece cada observacion.
    Post: Devuelve las metricas del modelo de clasificación
"""
def get_metrics_model(model, x_train, y_train, x_test, y_test):
    return get_metrics(y_test, predict_values(model, x_train, y_train, x_test))

def predict_values(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    return model.predict(x_test)

def get_prob(model, x_train, y_train, x_test):
    return model.fit(x_train, y_train).predict_proba(x_test)

def got_roc_values_model(model, x_train, y_train, x_eval, y_eval):
    model.fit(x_train, y_train)
    yPred = model.predict(x_eval)
    return got_roc_values(y_eval, yPred)

def got_roc_values(x_data, y_data):
    fpr, tpr, thresholds = roc_curve(x_data, y_data)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

def got_roc_values_multiclass(num_classes, y_test, y_scores, clases_numbers):
    y_test_onehot = one_hot(y_test, num_classes)
    fpr = [0] * num_classes
    tpr = [0] * num_classes
    roc_auc = [0] * num_classes
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_onehot[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(num_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= num_classes

    return fpr_grid, mean_tpr, auc(fpr_grid, mean_tpr)

def got_zip_class_prob(class_list, prob_ist):
    zip_true_label_probs = []
    for x in range(len(class_list)):
        zip_true_label_probs += [(class_list[x], list(prob_ist[x]))]
    
    return zip_true_label_probs

def compare_models_bar_graph(x_train, reduced_train, y_train, 
                             x_eval, reduced_eval, y_eval, 
                             autoencoder, filename, number_classes):

    compare_model_bar_graph(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, KNeighborsClassifier(n_neighbors = number_classes), 
                            "KNN", filename)
    compare_model_bar_graph(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, DecisionTreeClassifier(), 
                            "DT", filename)
    compare_model_bar_graph(x_train, reduced_train, y_train,
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, RandomForestClassifier(), 
                            "RF", filename)
    compare_model_bar_graph(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3), 
                            "GB", filename)
    compare_model_bar_graph(x_train, reduced_train, y_train,
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, svm.SVC(kernel='linear'), 
                            "SVM", filename)
    
def compare_models_roc_curve(x_train, reduced_train, y_train, 
                             x_eval, reduced_eval, y_eval, 
                             autoencoder, filename, number_classes):
    
    compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, KNeighborsClassifier(n_neighbors = number_classes), 
                            "KNN", filename,number_classes)
    compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, DecisionTreeClassifier(), 
                            "DT", filename, number_classes)
    compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, RandomForestClassifier(), 
                            "RF", filename, number_classes)
    compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3), 
                            "GB", filename, number_classes)
    compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, svm.SVC(kernel='linear', probability = True), 
                            "SVM", filename, number_classes)

def compare_models_MAUC(x_train, reduced_train, y_train, 
                        x_eval, reduced_eval, y_eval, 
                        autoencoder, filename):
    
    compare_model_MAUC(x_train, reduced_train, y_train, 
                       x_eval, reduced_eval, y_eval, 
                       autoencoder, KNeighborsClassifier(n_neighbors=3), 
                       "KNN", filename)
    compare_model_MAUC(x_train, reduced_train, y_train, 
                       x_eval, reduced_eval, y_eval, 
                       autoencoder, DecisionTreeClassifier(), 
                       "DT", filename)
    compare_model_MAUC(x_train, reduced_train, y_train, 
                       x_eval, reduced_eval, y_eval,
                       autoencoder, RandomForestClassifier(), 
                       "RF", filename)
    compare_model_MAUC(x_train, reduced_train, y_train,
                       x_eval, reduced_eval, y_eval,
                       autoencoder, GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3), 
                       "GB", filename)
    compare_model_MAUC(x_train, reduced_train, y_train,
                       x_eval, reduced_eval, y_eval,
                       autoencoder, svm.SVC(kernel='linear', probability = True), 
                       "SVM", filename)
    
def plot_matrix(x_train, y_train, x_test, y_test, filename, using_DL, number_classes):
    labels = ["CN","MCI","AD"]
    if number_classes == 2 :
        labels = ["sMCI", "pMCI"]


    plot.plot_confusion_matrix(y_test, predict_values(KNeighborsClassifier(n_neighbors = number_classes), x_train, y_train, x_test), "KNN" + using_DL, labels)
    plot.plot_confusion_matrix(y_test, predict_values(RandomForestClassifier(), x_train, y_train, x_test), "RF" + using_DL,labels)
    plot.plot_confusion_matrix(y_test, predict_values(DecisionTreeClassifier(), x_train, y_train, x_test), "DT" + using_DL,labels)
    plot.plot_confusion_matrix(y_test, predict_values(GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3), 
                                            x_train, y_train, x_test), "GB" + using_DL, labels)
    plot.plot_confusion_matrix(y_test, predict_values(svm.SVC(kernel='linear'), x_train, y_train, x_test), "SVM" + using_DL, labels)



def show_result(x_train, y_train, x_eval, y_eval, filename, using_DL, number_classes):

    plot_matrix(x_train, y_train, x_eval, y_eval, filename, using_DL, number_classes)
    
    show_result_bar_graph(x_train, y_train, x_eval, y_eval, filename, using_DL, number_classes)
             
    show_result_roc_curves(x_train, y_train, x_eval, y_eval, filename, using_DL, number_classes)

def show_result_bar_graph(x_train, y_train, x_eval, y_eval, filename, using_DL, number_classes):
    # Obtenemos las metricas, los valores predecidos y el modelo ya entrenado
    knn_metrics = get_metrics_model(KNeighborsClassifier(n_neighbors = number_classes),
                                   x_train, y_train, 
                                   x_eval, y_eval)

    svm_metrics = get_metrics_model(svm.SVC(kernel='linear'),
                                   x_train, y_train, 
                                   x_eval, y_eval)

    dt_metrics = get_metrics_model(DecisionTreeClassifier(),
                                  x_train, y_train, 
                                  x_eval, y_eval)

    rf_metrics = get_metrics_model(RandomForestClassifier(),
                                  x_train, y_train, 
                                x_eval, y_eval)

    gb_metrics = get_metrics_model(GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3),
                                  x_train, y_train, 
                                  x_eval, y_eval)
    
    # Dividimos las metricas para cada uno
    accuracy = [knn_metrics[0], svm_metrics[0], dt_metrics[0], rf_metrics[0], gb_metrics[0]]
    precision = [knn_metrics[1], svm_metrics[1], dt_metrics[1], rf_metrics[1], gb_metrics[1]]
    recall = [knn_metrics[2], svm_metrics[2], dt_metrics[2], rf_metrics[2], gb_metrics[2]]
    f1_score = [knn_metrics[3], svm_metrics[3], dt_metrics[3], rf_metrics[3], gb_metrics[3]]
    x_ticks = ['kNN' + using_DL,'SVM' + using_DL,'Decision Trees' + using_DL, \
              'Random Forest' + using_DL, 'Gradient Boosting' + using_DL]
    
    # Ploteamos el grafico de barras
    plot.plot_bar_graph(5, 4, [accuracy, precision, recall, f1_score], ['r', 'g', 'b', 'y'],
                 ["Accuracy", "Precision", "Recall", "F1Score"], x_ticks, filename, using_DL)

def compare_model_bar_graph(x_train, reduced_train, y_train, 
                         x_eval, reduced_eval, y_eval, 
                         autoencoder, model, model_name, filename):
    
    model_baseline_metrics = get_metrics_model(clone(model),
                                 x_train, y_train, 
                                 x_eval, y_eval)
    
    model_reduce_metrics = get_metrics_model(clone(model),
                                 reduced_train, y_train, 
                                 reduced_eval, y_eval)
    
    autoencoder_metrics = autoencoder.get_metrics(autoencoder.predict_fine_tuning(x_eval, 0), y_eval)

    # Dividimos las metricas para cada uno
    accuracy = [model_baseline_metrics[0], autoencoder_metrics[0], model_reduce_metrics[0]]
    precision = [model_baseline_metrics[1], autoencoder_metrics[1], model_reduce_metrics[1]]
    recall = [model_baseline_metrics[2], autoencoder_metrics[2], model_reduce_metrics[2]]
    f1_score = [model_baseline_metrics[3], autoencoder_metrics[3], model_reduce_metrics[3]]
    x_ticks = [model_name,'Autoencoder', model_name + ' + DL']
    plot.plot_bar_graph(3, 4, [accuracy, precision, recall, f1_score ], ['r', 'g', 'b', 'y'],
                 ["Accuracy", "Precision", "Recall", "F1Score"], x_ticks, filename, "compare" + model_name)

def compare_model_roc_curve(x_train, reduced_train, y_train, 
                            x_eval, reduced_eval, y_eval, 
                            autoencoder, model, 
                            model_name, filename, number_classes):

    bsln_fpr, bsln_tpr, bsln_roc_auc = got_roc_values_multiclass(number_classes, y_eval, get_prob(clone(model),x_train, y_train, x_eval), list(range(0,number_classes)))

    red_fpr, red_tpr, red_roc_auc = got_roc_values_multiclass(number_classes, y_eval, get_prob(clone(model),reduced_train, y_train, reduced_eval), list(range(0,number_classes)))

    ae_fpr, ae_tpr, ae_roc_auc = got_roc_values_multiclass(number_classes, y_eval, autoencoder.predict_proba(x_eval,0), list(range(0,number_classes)))
    
    fpr_list = [bsln_fpr, ae_fpr, red_fpr]
    tpr_list = [bsln_tpr, ae_tpr, red_tpr]
    roc_auc_ist = [bsln_roc_auc, ae_roc_auc, red_roc_auc]
    model_list = ['Baseline ' + model_name + ' Mean (area = %0.3f)', 'Autoencoder (area = %0.3f)', model_name + ' + DL (area = %0.3f)']

    plot.plot_roc_curves(3,fpr_list, tpr_list, roc_auc_ist, model_list,  "Macro Average", "compare" + model_name)

def compare_model_MAUC(x_train, reduced_train, y_train, 
                     x_eval, reduced_eval, y_eval, 
                     autoencoder, model, 
                     model_name, filename):

    bsl_MAUC = MAUC(got_zip_class_prob(y_eval, get_prob(clone(model),x_train, y_train, x_eval)) , 3)

    red_MAUC = MAUC(got_zip_class_prob(y_eval, get_prob(clone(model),reduced_train, y_train, reduced_eval)) , 3)

    ae_MAUC = MAUC(got_zip_class_prob(y_eval, autoencoder.predictProbs(x_eval,0)), 3)

    MAUC_list = [bsl_MAUC, ae_MAUC, red_MAUC]
    x_ticks = [model_name,'Autoencoder', model_name + ' + DL']

    plot.plot_bar_graph(3, 2, [[], MAUC_list] , ['w', 'b'], ["", 'MAUC'], x_ticks, filename, "compareMAUC" + model_name)



def show_result_roc_curves(x_train, y_train, x_eval, y_eval, 
                           filename, using_DL, number_classes):

    knn_fpr, knn_tpr, knn_roc_auc = got_roc_values_multiclass(number_classes, y_eval, 
                                                              get_prob(KNeighborsClassifier(n_neighbors = number_classes),
                                                                       x_train, y_train, x_eval),
                                                              list(range(0,number_classes)))

    svm_fpr, svm_tpr, svm_roc_auc = got_roc_values_multiclass(number_classes, y_eval, 
                                                              get_prob(svm.SVC(kernel='linear', probability = True),
                                                                       x_train, y_train, x_eval),
                                                              list(range(0,number_classes)))

    dt_fpr, dt_tpr, dt_roc_auc = got_roc_values_multiclass(number_classes, y_eval, 
                                                           get_prob(DecisionTreeClassifier(),
                                                                    x_train, y_train, x_eval),
                                                           list(range(0,number_classes)))

    rf_fpr, rf_tpr, rf_roc_auc = got_roc_values_multiclass(number_classes, y_eval, 
                                                           get_prob(RandomForestClassifier(),
                                                                    x_train, y_train, x_eval),
                                                           list(range(0,number_classes)))

    gb_fpr, gb_tpr, gb_roc_auc = got_roc_values_multiclass(number_classes, y_eval, 
                                                           get_prob(GradientBoostingClassifier(n_estimators = 20, learning_rate = 0.3),
                                                                    x_train, y_train, x_eval),
                                                           list(range(0,number_classes)))
    fpr_list = [knn_fpr, svm_fpr, dt_fpr, rf_fpr, gb_fpr]
    tpr_list = [knn_tpr, svm_tpr, dt_tpr, rf_tpr, gb_tpr]
    roc_auc_list =[knn_roc_auc, svm_roc_auc, dt_roc_auc, rf_roc_auc, gb_roc_auc]
    model_list = ['KNN ' + using_DL + ' (area = %0.3f)', 
                 'SVM ' + using_DL + '  linear (area = %0.3f)', 
                 'Tree ' + using_DL + '  (area = %0.3f)', 
                 'RF ' + using_DL + '  (area = %0.3f)', 
                 'GB ' + using_DL + '  (area = %0.3f)']
    
    plot.plot_roc_curves(5,fpr_list, tpr_list, roc_auc_list, model_list, "Macro Average", using_DL)


def evaluate_AE_with_shallow(x_train, x_eval, y_train, y_eval,
                             learning_rate, learning_rate_ft, 
                             batch_size, epochs, dropout_value,
                             size_layers_used, latent_space,
                             encoder_func, decoder_func, verbose,
                             text, number_classes):
    
    autoencoder = train_autoencoder(x_train, y_train, 
                                   learning_rate, learning_rate_ft, 
                                   batch_size, epochs, dropout_value, 
                                   size_layers_used, latent_space, 
                                   encoder_func, decoder_func, verbose, 1, number_classes)

    print(autoencoder.get_metrics(autoencoder.predict_fine_tuning(x_eval, 0), y_eval))


    #MAUC values
    # Devolvemos los datos reducidos mediante el autoencoder
    reduced_train = autoencoder.return_reduce_attribute(x_train)
    reduced_eval = autoencoder.return_reduce_attribute(x_eval)

    compare_models_roc_curve(x_train, reduced_train, y_train, 
                          x_eval, reduced_eval, y_eval, autoencoder, "TADPOLE D4", number_classes)
    compare_models_bar_graph(x_train, reduced_train, y_train, 
                          x_eval, reduced_eval, y_eval, autoencoder, "TADPOLE D4", number_classes)
    show_result(x_train, y_train, x_eval, y_eval,  "TADPOLE D4", "", number_classes)
    show_result(reduced_train, y_train, reduced_eval, y_eval, "TADPOLE D4", " + DL", number_classes)

def main(argv):

    # Si los argumentos son correctos, se anyadiran los features a tener en cuenta
    features =  read_file("features/others")
    AD_or_MCI = 0
    text = "CN/MCI/AD problem"
    using_DX = 1
    train_data_Path = "TrainTadpole.csv"
    eval_data_Path = "EvalTadpole.csv"
    feature_type_path = "Feature_Type.csv"
    number_classes = 3
    clinicPaths= ["Data/TADPOLE_D1_D2.csv", "Data/TADPOLE_D3.csv", "Data/TADPOLE_D4_corr.csv"]
    for i in range(len(argv)):
        if (argv[i] == "-C"):
            features = features + read_file("features/cognitive")

        elif (argv[i] == "-MRI"):
            features = features + read_file("features/UCSFFSL")
            features = features + read_file("features/UCSFFSX")
           
        elif (argv[i] == "-PET"):
            features = features + read_file("features/BAIPETNMRC")
            features = features + read_file("features/UCBERKELEYAV45-1451")
            
        elif (argv[i] == "-DTI"):
            features = features + read_file("features/DTIROI")

        elif (argv[i] == "-BIO"):
            features = features + read_file("features/Biomarkers")

        elif (argv[i] == "-sMCIpMCI"):
            AD_or_MCI = 1
            text = "sMCI/pMCI"
            number_classes = 2
        
        elif (argv[i] == "-NoDX"):
            using_DX = 0

        elif (argv[i] == "-DELETE"):
            if os.path.exists(train_data_Path):
                os.remove(train_data_Path)
            if os.path.exists(eval_data_Path):
                os.remove(eval_data_Path)
        
    # Error por pantalla al usar mal el comando de ejecucion del programa
    if (len(argv) > 8):
        usage = "py main.py [-C] [-MRI] [-PET] [-DTI] [-BIO] [-sMCIpMCI] [-DELETE] [-NoDX]"

        print("Usage: " + usage, file=sys.stderr)
        exit()

    trainData, evalData = [], []
    dataSet = Datasets()
    # Creamos el dataset de los datos médicos y prepocesamos los datos
    [trainData, evalData] = dataSet.loadTADPOLE(clinicPaths, features, feature_type_path, AD_or_MCI, using_DX)
    

    [xTrain, yTrain] = dataSet.divideData(trainData)
    [xEval, yEval] = dataSet.divideData(evalData)
    
    sizeLayersUsed = [500,300]
    latent_space = 150

    #hyperParametersAE(xTrain, yTrain, sizeLayersUsed, latent_space, "relu", "relu")
    #exit()
    evaluate_AE_with_shallow(xTrain, xEval, yTrain, yEval,
                           LEARNINGRATE, LEARNINGRATEFT,
                           BATCHSIZE, EPOCHS, DROPOUTVALUE,
                           sizeLayersUsed,  latent_space, 
                           "relu", "relu", 1, text, number_classes)

if __name__ == "__main__":
    main(sys.argv[1:])
    