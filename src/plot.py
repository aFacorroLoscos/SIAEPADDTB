from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Funciones plot
def plot_bar_graph(n, number_metrics, metric_list, color_metric, label_metric, 
    x_ticks, dest_folder, suffix):
    """
    Genera un grafico de barras y lo guarda en la carpeta destino

    :param n: Numero de conjuntos de barras
    :param number_metrics: Numero de barras unidas 
    :param metric_list: Lista de metricas para generar las barras
    :param color_metric: Lista de colores para las barras
    :param label_ metric: Lista de metricas para generar leyenda
    :param x_ticks: Lista de nombres para generar en la direccion de la x
    :param dest_folder: Carpeta destino
    :param suffix: Sufijo que se anyade a la carpeta destino
    """
    #Plot Bar graph
    ind = np.arange(n)
    width = 0.1
    plt.figure().set_figwidth(10)
    for i in range(number_metrics):
        if metric_list[i]:
           plt.bar(ind + width * i, metric_list[i], width, color = color_metric[i], label = label_metric[i])

    plt.ylabel("Performance Metrics")
    plt.xticks(ind + width, x_ticks)
    plt.legend(loc ="lower right")
    plt.ylim(0.8,1.0)
    plt.savefig("../results/" + dest_folder + "/barGraph" + suffix + '.png', bbox_inches='tight')
    plt.close()

def plot_roc_curves(number_models, fpr_list, tpr_list, roc_auc_list, model_list, dest_folder, suffix):
    """
    Genera una curva ROC segun el valor de los parametros y la guarda en una carpeta destino

    :param number_models: Numero de modelos
    :param fpr_list: Lista de false positive rate
    :param tpr_list: Lista de true positive rate
    :param roc_auc_list: Lista de valores roc auc
    :param model_list metric: Lista de modelos
    :param dest_folder: Carpeta destino
    :param suffix: Sufijo que se anyade a la carpeta destino
    """
    # Plot ROC curve
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for i in range(number_models):
        plt.plot(fpr_list[i], tpr_list[i], label= model_list[i] % roc_auc_list[i])
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.savefig("../results/" + dest_folder + "/rocCurves" + suffix + '.png', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_data, y_pred, dest_folder, suffix, labels):

    """
    Genera una matriz de confusion segun y_data e y_pred y la guarda en la carpeta destino

    :param y_data: Contiene la clase a la que pertenece cada observacion
    :param y_pred: Contiene la clase de cada observacion que ha predicho el modelo
    :param dest_folder: Carpeta destino
    :param suffix: Sufijo que se anyade a la carpeta destino
    :param labels: Lista que contiene el string de los nombres de las clases
    """
    # Crear la matriz de confusión utilizando sklearn
    cm = confusion_matrix(y_data, y_pred)

    # Crear la visualización de la matriz de confusión
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

    # Graficar la matriz de confusión
    cm_display.plot()
    plt.savefig("../results/" + dest_folder + "/confusionMatrix" + suffix + '.png', bbox_inches='tight')
    plt.close()
