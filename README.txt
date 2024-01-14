-------------------------------------------------------
	   REQUISITOS Y VERSIONES DE MODULOS
-------------------------------------------------------
	Pyton 		-> 3.10
	Numpy		-> 1.22
	Pandas		-> 1.5
	Matplotlib	-> 3.6
	Sklearn 	-> 0.0
 	Tensorflow 	-> 2.10
	Keras		-> 2.10


-------------------------------------------------------
		FUNCIONAMIENTO DEL PROGRAMA
-------------------------------------------------------
El programa consiste en un modelo de redes neuronales llamado Autoencoder, dicho modelo se encarga
de aprender a codificar la entrada de datos y luego generar datos de salida lo mas proximos a la entrada dada.
El conjunto de datos se ha obtenido del TADPOLE challenge, siendo el fichero TADPOLE_D1_D2.csv el conjunto de
datos que se utilizarán en la etapa Train y Test de nuestro modelo y el fichero TADPOLE_D4_corr.csv contendrá los datos para la etapa evaluacion
Ya que ambos conjuntos de datos son diferentes, se ha optado por tener en cuenta aquellos pacientes que estan en el fichero D4 y que a la vez estan en el fichero
D1-D2 para la fase de evaluacion, dichos pacientes se han eliminado del fichero D1-D2 para evitar data leak.

El programa tiene dos partes:
En la primera parte se extrae los datos de los ficheros csv el cual contiene datos de tipo: MRI, PET, DTI, CSF, Geneticos,
test cognitivos y datos demográficos entre otros.
Una vez extraido el fichero se realiza un pre-procesado al archivo en el cual eliminaremos aquellos atributos que tengan un MISSING RATE
mayor a 70, tambien transformaremos
aquellos valores cualitativos a cuantitativos mediante ONE HOT ENCODING y por último los valores cuantitativos se normalizaran entre 1 y 2. 
Algunos atributos como las fechas, timelapses o identificadores no se tienen en cuenta ya que no aportan informacion relevante.

La segunda parte es el autoencoder: Dividimos el dataframe procesado en un conjunto train y test según un 90/10%.
El conjunto train se utiliza para entrenar el autoencoder con una serie de hyperparametros ya dados que se han
obtenido mediante el uso de K-fold cross validation, tambien se utilizan para realizar el proceso de FINE TUNING en el que solo nos quedamos con la capa ENCODER
del autoencoder y le añadimos una capa softmax que devolvera la probabilidad de pertenecer a una de las clases de los datos. 
Tras el proceso de entrenamiento, eliminaremos la capa de softmax y el modelo devolvera los datos train
y test reducidos al tamaño de la ultima capa del ENCODER para que estos sean utilizados por un modelo de clasificacion como: 
KNN, SVM, Random Forest, Decision Tree y Gradient Boosting.
Una vez entrenados y testeados se generará un grafica de barras con el valor de métricas (Accuracy, Precision, Recall,
F1 SCORE) de cada uno de los modelos.
Tambien se generará un gráfico de barras sin el uso del Autoencoder para predecir las clases utilizando los modelos
de clasificación anteriormente nombrados para comparar resultados entre ambos gráficos.

En la carpeta Resultados se encuentran las imagenes de los resultados obtenidos

En el fichero Features_eliminado.txt estan todos los features que se han eliminado debido a que eran datos de tipo fecha
o timelapse

-------------------------------------------------------
		EJECUCION DEL PROGRAMA
-------------------------------------------------------
Para ejecutar el programa se tendrá que escribir el siguiente comando:
python3 main.py [-C] [-MRI] [-PET] [-DTI] [-BIO] [-sMCIpMCI] [-DELETE] [-NoDX]

Al añadirse las opciones -C, -MRI, -PET, -DTI, -BIO indicaremos que ese tipo de dato quiere tenerse en cuenta en el preprocesado, entrenamiento y evaluacion

Si añadimos la opcion -sMCIpMCI tendremos en cuenta el problema sMCI vs pMCI en vez de CN vs MCI vs AD a la hora de elegir el diagnostico

En caso de añadir la opcion -NoDX no tendremos en cuenta la columna DX en la fase de Procesamiento de los datos

La Opcion -DELETE sirve para borrar los ficheros de TADPOLE que generamos en la fase de Procesamiento