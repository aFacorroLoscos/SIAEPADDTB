from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Funciones plot

def plot_bar_graph(n, number_metrics, metric_list, color_metric, label_metric, 
                   x_ticks, dest_folder, suffix):
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
    plt.ylim(0,1.0)
    plt.savefig("../results/" + dest_folder + "/barGraph" + suffix + '.png', bbox_inches='tight')
    plt.close()

def plot_roc_curves(number_models, fpr_list, tpr_list, roc_auc_list, model_list, dest_folder, suffix):
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
    # Crear la matriz de confusión utilizando sklearn
    # Crear la matriz de confusión utilizando sklearn
    cm = confusion_matrix(y_data, y_pred)

    # Crear la visualización de la matriz de confusión
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = labels)

    # Graficar la matriz de confusión
    cm_display.plot()
    plt.savefig("../results/" + dest_folder + "/confusionMatrix" + suffix + '.png', bbox_inches='tight')
    plt.close()
