Estudio de la influencia del autoencoder en el
problema de diagnóstico de la enfermedad de
Alzheimer a partir de datos tabulares
======================

![alt text]([https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true](https://github.com/aFacorroLoscos/SIAEPADDTB/blob/main/model_proyect.png))


Bibliotecas utilizadas y versiones de las mismas
---------------------

| Biblioteca     | Version    |
|----------------|------------|
|Python          |3.10        |
|Nympy           |1.24.3      |
|Pandas          |1.5	      |
|Matplotlib      |3.6	      |
|Sklearn         |0.0.post5   |
|Tensorflow      |2.10	      |
|Keras		 |2.10	      |

Funcionamiento del Programa
-------------------------------------------------------
El programa consiste en el uso de un modelo de redes neuronales llamado Autoencoder, dicho modelo se encarga
de aprender a codificar y decodificar la entrada de datos para luego generar datos de salida lo mas proximos a la entrada dada.

El conjunto de datos se ha obtenido del TADPOLE challenge, siendo el fichero TADPOLE_D1_D2.csv el conjunto de
datos que se utilizarán en la etapa Train y Test de nuestro modelo y el fichero TADPOLE_D4_corr.csv contendrá los datos para la etapa evaluacion
Ya que ambos conjuntos de datos tienen un número diferente de atributos se ha optado por tener en cuenta aquellos pacientes que estan en el fichero D4 y que a la vez estan en el fichero
D1-D2 para la fase de evaluacion, dichos pacientes se han eliminado del fichero D1-D2 para evitar data leak.

El programa tiene dos partes:
En la primera parte se extrae los datos de los ficheros csv el cual contiene datos de tipo: MRI, PET, DTI, CSF, Geneticos,
test cognitivos y datos demográficos entre otros.
Una vez extraido el fichero se realiza un pre-procesado al archivo en el cual eliminaremos aquellos atributos que tengan un MISSING RATE
mayor a 70, tambien transformaremos aquellos valores cualitativos a cuantitativos mediante ONE HOT ENCODING y por último los valores cuantitativos se normalizaran a valores entre 1 y 2. 
Algunos atributos como las fechas, timelapses o identificadores no se tienen en cuenta ya que no aportan informacion relevante.

Dependiendo del problema a tener en cuenta CN-MCI-AD o sMCI-pMCI el diagnóstico de los datos cambiará:
En caso de procesar los datos al problema CN-MCI-AD el diagnóstico dependerá del valor
DXCHANGE, cambiando el diagnóstico entre consultas al diagnóstico CN-MCI-AD.
Si se quiere procesar los datos al problema sMCI-pMCI solo se tendrán en cuenta los diagnósticos MCI de un periodo de 3 años con respecto a la primera consulta, si en 
esos 3 años alguno de los pacientes sufre un cambio MCI a AD, entonces es un caso de pMCI, sMCI en el caso de ser diagnosticado con MCI en ese periodo de 3 años.

Tras procesar los datos el programa creará dos ficheros con formato .csv donde se encuentra el conjunto de datos Train y conjunto de datos Eval.

La segunda parte depende del tipo de ejecución que queremos:
Se puede realizar una evaluación de los hyper-parámetros del modelo Autoencoder mediante la ejecución del algoritmo K fold cross validation para obtener 
los mejores valores de los diferentes hyper-parámetros del Autoencoder.

Podemos comparar el modelo Autoencoder con un modelo basado en redes neuronales y comprobar las métricas que se obtienen al evaluarlos con el conjunto de datos de evaluación.

Por último, comparamos el modelo Autoencoder con la ejecución de modelos de aprendizaje tradicional y con la ejecucición de modelos de aprendizaje tradicional en combinacion
con el modelo Autoencoder. En primer lugar el programa se encarga de entrenar el modelo con el conjunto de datos de entrenamiento dados, el  entrenamiento
del autoencoder se realiza con una serie de hyperparametros ya dados que se han obtenido mediante el uso de K-fold cross validation, tambien se utilizan para realizar 
el proceso de FINE TUNING en el que solo nos quedamos con la capa ENCODER del autoencoder y le añadimos una capa softmax que devolvera la probabilidad de pertenecer a una de las clases de los datos. 

Tras el proceso de entrenamiento, eliminaremos la capa de softmax y el modelo devolvera los datos train
y test reducidos al tamaño de la ultima capa del ENCODER para que estos sean utilizados por un modelo de clasificacion tradicional como: KNN, SVM, Random Forest, Decision Tree y Gradient Boosting.
Una vez entrenados y testeados se generará un grafica de barras con el valor de métricas (Accuracy, Precision, Recall,
F1 SCORE) de cada uno de los modelos, también se generarán gráficas curvas ROC para evaluar los distintos modelos
Tambien se generará un gráfico de barras con el uso y sin el uso del Autoencoder para reducir el conjunto de datos utilizando solo los modelos
de clasificación anteriormente nombrados para comparar resultados entre ambos gráficos.


Carpetas del Proyecto
-------------------------------------------------------
En la carpeta results se encuentran las imagenes de los resultados obtenidos dependiendo de si el problema es CN-MCI-AD o sMCI-pMCI.

En el fichero Features_eliminado.txt estan todos los features que se han eliminado debido a que eran datos de tipo fecha
o timelapse y por tanto no eran necesarios en el proyecto.

La carpeta features se encuentran los distintos features utilizados para procesar los datos, dichos features estan divididos segun el tipo de pruebas
de las cuales se han obtenido ese tipo de valores.

En la carpeta data se deberá introducir el conjunto de datos TADPOLE para así poder procesar el conjunto de datos.


Ejecucion del programa en línea de comandos
-------------------------------------------------------
Para ejecutar el programa se tendrá que escribir los siguientes conjuntos de comandos:

En caso de generar los ficheros de datos procesados se deberá ejecutar el siguiente formato:
```
$ python3 main.py -GENERATE [[-C] [-MRI] [-PET] [-DTI] [-BIO] [-ALL] [-sMCIpMCI] [-DELETE] [-useDX]
```
Al añadirse las opciones: `-C`, `-MRI`, `-PET`, `-DTI`, `-BIO`. Se tendrán en cuenta un conjunto de features en el procesamiento de los datos.

Si añadimos la opcion `-sMCIpMCI` tendremos en cuenta el problema sMCI vs pMCI en vez del problema CN vs MCI vs AD a la hora de elegir el diagnostico.

En caso de utilizar la opcion `-NoDX` no tendremos en cuenta la columna DX, DXCHANGE y DX_bl en la fase de Procesamiento de los datos.

La Opcion `-DELETE` sirve para borrar los ficheros de TADPOLE que generamos en la fase de Procesamiento en caso de querer sustituirlos por
otros nuevos.


En caso de entrenar modelos y evaluarlos se deberá ejecutar el siguiente formato:	
```
$ python3 main.py -LOAD pathTrainData pathTestData [-COMPARE] [-KFOLD] [-sMCIpMCI]
```
pathTrainData pathTestData son los paths correspondientes a los ficheros procesados .csv del conjunto de Entrenamiento y conjunto de Evaluación ya procesados
anteriormente.

Este comando tiene tres maneras diferentes de ejecución dependiendo de cual de los dos siguientes flags se active:
Si se activa `-COMPARE` se ejecutará una comparación entre modelo Neuronal Network y Autoencoder mostrando las métricas entre ambos modelos tras ser
entrenados con el conjunto de datos Train y evaluados con el conjunto de datos Eval.

Si se quiere comprobar que hyper-parámetros son los que mejores resultados da con el conjunto de datos se deberá activar el flag `-KFOLD`.

En caso de no activar ninguno de los dos anteriores flags nombrados, realizaremos la ejecución de comparar el modelo Autoencoder-based junto a los modelos
de aprendizaje tradicionales ya nombrado anteriormente.

Por último el flag `-sMCIpMCI` se activa en caso de que el conjunto de variables y corresponda al problema de sMCIpMCI.
