import sys
import os
from types import NoneType # Para poder usar el tipo None
import pandas as pd # Pandas para los dataflow
import re # Para expresiones regulares
import numpy as np # Acceder a datos np.int o np.float
from sklearn.impute import KNNImputer
from itertools import repeat # Para funcion repeat
from contextlib import redirect_stdout # Redirigir salida a fichero
from pathlib import Path
import logging
import math

class Datasets:

    """
        Pre: ---
        Post: Inicializamos la clase Dataset
    """
    def __init__( self ):
        self.columns = ['RID', 'AGE']

        self.excludedColumns = ["RID", "PTID", "Diagnosis", "DXCHANGE"]

        self.d1d2DataFrame = None
        self.d3DataFrame = None
        self.d4DataFrame = None
        self.ADNIDataframe = None


    # --------------------------------------------------------------------------------------

    """
        Pre: valueToPrint es un vector de valores, filename es un path a un archivo
        Post: Escribe valueToPrint en el fichero definido en filename
    """
    def printListIntoFile(self, valueToPrint, filename):
        with open(filename + '.txt', 'w') as f:
            with redirect_stdout(f):
                print('\n'.join(map(str, valueToPrint)))

    # --------------------------------------------------------------------------------------

    """
        Pre: value y maxValue es un valor numerico 
        Post: Normaliza el valor entre 1 y 2
    """
    def normalizedValues(self, value, maxValue, minValue):
        result = np.nan
        if not pd.isna(value):
            result = '{0:.3f}'.format((value - minValue) / (maxValue - minValue))
        return result

    """
        Pre: value y maxValue es un valor numerico 
        Post: Normaliza el valor entre 1 y 2
    """
    def standarizedValues(self, value, mean, std):
        result = 0
        if not pd.isna(value):
            result = '{0:.3f}'.format((value - mean) / std)
        return result

    # --------------------------------------------------------------------------------------
    
    """
        Pre: _table debe ser un dataframe de datos
        Post: Devolvemos un dataframe de datos que no tengan las filas de aquellos valores
                de la columna RID y VISCODE que queremos censurar.
    """
    def censor_d1_table(self, _table):
        # TANTO RID COMO VISCODE SON COLUMNAS DEL CONJUNTO DE DATOS TADPOLE
        _table.loc[(_table.RID == 2190) & (_table.VISCODE == 'm03')]
        _table.drop(_table.loc[(_table.RID == 2190) & (_table.VISCODE == 'm03')].index, inplace=True)
        _table.drop(_table.loc[(_table.RID == 4579) & (_table.VISCODE == 'm03')].index, inplace=True)
        _table.drop(_table.loc[(_table.RID == 1088) & (_table.VISCODE == 'm72')].index, inplace=True)
        _table.drop(_table.loc[(_table.RID == 1195) & (_table.VISCODE == 'm48')].index, inplace=True)
        _table.drop(_table.loc[(_table.RID == 4960) & (_table.VISCODE == 'm48')].index, inplace=True)
        # _table.drop(_table.loc[(_table.RID == 4674)].index, inplace=True)
        _table.drop(_table.loc[(_table.RID == 5204)].index, inplace=True)

        # PLOS One data
        _table.drop(_table.loc[(_table.RID == 2210) & (_table.VISCODE == 'm60')].index, inplace=True)


        _table.reset_index()    

        # --------------------------------------------------------------------------------------

    """
        Pre: train_df es un dataframe
        Post: Llenamos el dataframe con valores antiguos
    """
    def fill_diagnans_by_older_values(self, train_df):
        """Fill nans in Diagnosis column from feature matrix by older values (ffill), then by newer (bfill)"""
        
        df = train_df.copy()
        
        # Ordenacion EMC-EB

        train_df['AUX'] = train_df['AGE']
        
        train_df['AUX'] += train_df['Month_bl'] / 12. # Cuidado no este sumando dos veces
        train_df = train_df.sort_values(['RID','AUX'])
        
        # Ordenacion mhg
        #train_df = train_df.sort_values(['RID','VISCODE']) # Short values por VISCODE puede fallar en casos como m108

        df_filled_nans = train_df.groupby('RID')['DXCHANGE'].fillna(method='ffill')
        train_df['DXCHANGE'] = df_filled_nans
    
        df_filled_nans = train_df.groupby('RID')['DXCHANGE'].fillna(method='bfill')
        train_df['DXCHANGE'] = df_filled_nans

        df_filled_nans = train_df.groupby('RID')['DX'].fillna(method='ffill')
        train_df['DX'] = df_filled_nans
    
        df_filled_nans = train_df.groupby('RID')['DX'].fillna(method='bfill')
        train_df['DX'] = df_filled_nans
        
        train_df = train_df.drop(['AUX'], axis=1)

        return train_df

    # --------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------

    """
        Pre: adniD1D2Path es un path valido hacia el conjunto de datos TADPOLE D1-D2
        Post: Se devuelve un dataframe valido para el prepocesamiento de los datos
    """
    def loadTADPOLE(self, medicalPaths, features, featureTypePath, problemType, usingDX):
        TrainDataPath = "../TrainTadpole.csv"
        EvalDataPath = "../EvalTadpole.csv"
        if not os.path.exists(TrainDataPath) or not os.path.exists(EvalDataPath):
            # Load D1_D2 evaluation data set
            D1D2Path = Path(medicalPaths[0])
            D1D2df = pd.read_csv(D1D2Path, sep = ",", decimal=".", float_precision='high')              # [12741 rows x 1907 columns]

            #D3Path = Path(medicalPaths[1])
            #D3df = pd.read_csv(D3Path, sep = ",", decimal=".", float_precision='high')
                    
            D4Path = Path(medicalPaths[2])
            D4df = pd.read_csv(D4Path, sep = ",", decimal=".", float_precision='high')

            print( "Censoring D1D2 dataframe..." )
            self.censor_d1_table(D1D2df)
            print("D1D2 dataframe censored.")
            
        
            # Fill DXCHANGE nans by older values
            D1D2df = self.fill_diagnans_by_older_values( D1D2df )
            # Borrando indices que el valor DXCHANGE es NAN  
            print( "Delete dataframe's index with DXCHANGE NaN...")
            idx = D1D2df[np.isnan(D1D2df.DXCHANGE)].index
            D1D2df = D1D2df.drop(idx)
            print( "Delete done.")
            
            D1D2df = D1D2df[features]

            # PREPROCESS STAGE
            if problemType :
                D1D2df = D1D2df.loc[((D1D2df['DX_bl'] == "LMCI") | (D1D2df['DX_bl'] == "EMCI")) & (D1D2df['DX'] != "NL") & (D1D2df['DX'] != "MCI to NL")]
                D1D2df = self.sMCIpMCIDiagnosisTADPOLE(D1D2df)
            else: 
                D1D2df = self.obtainDiagnosisTADPOLE(D1D2df)

            if(not usingDX):
                D1D2df.drop(["DX_bl"], axis=1, inplace=True)
                D1D2df.drop(["DXCHANGE"], axis=1, inplace=True)
                D1D2df.drop(["DX"], axis=1, inplace=True)
            
            D1D2df.drop(["EXAMDATE"], axis=1, inplace=True)
            
            self.d1d2DataFrame = self.preprocess(D1D2df, featureTypePath)

            # TRAIN AND TEST DATAFRAME NO DATA LEAKAGE USING D3 AS EVAL AND D1-D2 AS TRAIN-TEST
            trainData = self.d1d2DataFrame.copy()
            evalData = self.d1d2DataFrame.copy()

            evalData = evalData[evalData['RID'].isin(D4df['RID'].unique())]
            
            trainData = trainData[~trainData['RID'].isin(evalData['RID'])]
            
            trainData.drop(["RID"], axis=1, inplace=True)
            evalData.drop(["RID"], axis=1, inplace=True)

            trainData.to_csv(TrainDataPath, index=False, sep = ";", float_format="%.3f")
            evalData.to_csv(EvalDataPath, index=False, sep = ";", float_format="%.3f")
            
            return [trainData, evalData]

        else:
            trainData = pd.read_csv(TrainDataPath, sep = ";")
            evalData = pd.read_csv(EvalDataPath, sep = ";")

            return [trainData, evalData]

    
    def obtainDiagnosisTADPOLE(self, dataframe):
        if 'Diagnosis' not in dataframe.columns:
            if 'DXCHANGE' in dataframe.columns:
                """We want to transform 'DXCHANGE' (a change in diagnosis, in contrast
                to the previous visits diagnosis) to an actual diagnosis."""
            
                # 7 -> 1 # 9 -> 1
                # 4 -> 2 # 8 -> 2
                # 5 -> 3 # 6 -> 3

                # Stable: Mantener valor, Conv: Pasar a una enfermedad mayor
                # Rev: Revertir la enfermedad a una etapa anterior 

                # 1 = Stable:NL to NL, 2 = Stable:MCI to MCI, 3=Stable:AD to AD
                # 4 = Conv:NL to MCI,  5 = Conv:MCI to AD,    6=Conv:NL to AD
                # 7 = Rev:MCI to NL,   8 = Rev:AD to MCI,     9=Rev:AD to NL
                # -1 = Not available
            
                # 1 es CN
                # 2 es MCI
                # 3 es AD

                # Creamos una nueva columna llamada Diagnosis y convertimos los 
                # valores DXCHANGE segun lo estipulado arriba

                dataframe['Diagnosis'] = dataframe['DXCHANGE']
                dataframe = dataframe.replace({'Diagnosis': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})

        dataframe["Diagnosis"] = dataframe["Diagnosis"] - 1
        return dataframe



    def sMCIpMCIDiagnosisTADPOLE(self, dataframe): 
        PTIDList = dataframe["PTID"].unique()

        resultDataframe = pd.DataFrame()

        for patientID in PTIDList:
            # Nos quedamos con solo los pacientes con el mismo PTID
            patientDiagnosis = dataframe.loc[dataframe['PTID'] == patientID].copy()

            # Solo tenemos en cuenta para el nuevo diagnostico sMCI o pMCI aquellos con un plazo de 3 años entre consultas
            patientDiagnosis['EXAMDATE'] = pd.to_datetime(patientDiagnosis["EXAMDATE"], format='%Y-%m-%d', errors='coerce')
            minimun_year = patientDiagnosis['EXAMDATE'].min() 
            

            patient_3_years = patientDiagnosis.loc[patientDiagnosis["EXAMDATE"] <= minimun_year + np.timedelta64(3,'Y')].copy()
            
            # Si se cuentra algún paciente con DEMENTIA or MCI to DEMENTIA entonces es Diagnostic = 1, 0 en caso contrario
            if len(patient_3_years.loc[((patient_3_years.DXCHANGE == 5) | (patient_3_years.DXCHANGE == 3))]) == 0:
                patient_3_years["Diagnosis"] = 0
            else:
                patient_3_years["Diagnosis"] = 1

            # Lo guardamos en un dataframe
            resultDataframe = pd.concat([resultDataframe, patient_3_years])
        
        resultDataframe[["PTID", "Diagnosis", "DXCHANGE", "DX", "DX_bl"]].to_csv("prueba.csv", index=False, sep = ";", float_format="%.3f")

        return resultDataframe
    
    """
        Pre: ---
        Post: Realiza el procesamiento del datafremo
    """
    def preprocess(self, dataframe, featurePath):

        print("Pre-processing...")

        df = dataframe.copy()
        # Adds months to age
        if 'Month_bl' in df.columns:
            df['AGE'] += df['Month_bl'] / 12. # Cuidado no este sumando dos veces
            df = df.drop(["Month_bl"], axis=1)
        

        # Convertimos la tabla DX y DX_bl que contienen string a numeros
        # Para facilitar el one hot encoding
        if 'DX_bl' in df.columns and type(df['DX_bl'].iloc[0]) is str:
            df = df.replace({'DX_bl': {'CN': 1, 'AD': 3, 'EMCI': 2, 'LMCI': 2, 'SMC': 2 }})
        if 'DX' in df.columns and type(df['DX'].iloc[0]) is str:   
            df = df.replace({'DX': {'Dementia': 3, 'Dementia to MCI': 8, 'MCI': 2, 'MCI to Dementia': 5, 'MCI to NL': 7, 'NL': 1, 'NL to MCI': 4 }})
            df = df.replace({'DX': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})    

        # Reemplazamos los valores vacios por NAN, los -4 tambien son NAN
        df = df.replace(' ', np.nan, regex=True)    
        df = df.replace('', np.nan, regex=True)
        df = df.replace("-4", -4, regex=True)
        df = df.replace(-4, np.nan, regex=True)
        
        dfQuant = pd.DataFrame()
        dfCat = pd.DataFrame()
        
        dataTypeDf = pd.read_csv(Path ("../Feature_Type.csv"), sep=';')
        totalValues = len(df.index)

        for (columnName, columnData) in df.items():
            # Columnas que no queremos tener en cuenta
            if(columnName.strip() not in self.excludedColumns):
                dataRow = dataTypeDf.loc[dataTypeDf["COLUMN"] == columnName]["TYPE"]

                # Columnas de tipo string
                if(columnName in dataTypeDf["COLUMN"].values and dataRow.values[0] == "T" ):
                    
                    if(columnData.isnull().mean() < 0.70 or columnName == "PTETHCAT"):
                        # Rellenamos la columna con datos por defecto
                        columnData = columnData.fillna(value='None')
                        # Realizamos el proceso de one hot encoding
                        oneHot = pd.get_dummies(columnData, columnName, drop_first=True, dtype=int)

                        if(df[columnName].isnull().values.any()):
                            oneHot = oneHot.drop([columnName + "_None"], axis = 1)

                        dfCat = pd.concat([dfCat, oneHot], axis = 1)
                        

                # Caso en que las columnas sean numericas
                elif(columnName in dataTypeDf["COLUMN"].values and dataRow.values[0] == "N" ):

                    if(columnData.isnull().mean() < 0.70):
                        columnData = pd.to_numeric(columnData)
                        # Convertimos todos los valores a valores numeros entre 1 y 2
                        minValue = np.nanmin(columnData)
                        maxValue = np.nanmax(columnData)
                        if(minValue < 0):
                                absValue = abs(minValue)
                                columnData = columnData + absValue 
                                minValue = np.nanmin(columnData)
                                maxValue = np.nanmax(columnData)

                        # mean = columnData.mean()
                        # std = columnData.std()
                        columnData = list(map(self.normalizedValues, columnData, repeat(maxValue), repeat(minValue)))
                        #columnData = list(map(self.standarizedValues, columnData, repeat(mean), repeat(std)))
                        newdf = pd.DataFrame(columns = [columnName])
                        newdf[columnName] = columnData
                        dfQuant = pd.concat([dfQuant, newdf], axis=1)
                # Caso en el que las columnas sean de un tipo desconocido
                else:
                    df.drop(columnName, axis=1, inplace=True)    


        imputer = KNNImputer(n_neighbors=10)

        result = pd.concat([df[["Diagnosis"]].reset_index(), dfCat.reset_index(), dfQuant.reset_index()], axis=1).drop("index", axis=1)

        result = pd.DataFrame(imputer.fit_transform(result),columns = result.columns)

        result = pd.concat([df[["RID"]].reset_index(), result.reset_index()], axis=1).drop("index", axis=1) #tex

        print("Pre-processing finished!")
        return result


    def droppedColumnsDataFrame(self, dataFrame):
        result = dataFrame.copy()
        for columnName in self.droppedColumns:
            if columnName in result.columns:
                result.drop([columnName], axis=1, inplace=True)
        return result

    """
        Pre: listOfList es un array de dos dimensiones de strings
        Post: Devolvemos la transformacion de un array de dos dimensiones de 
                string a un array de enteros 
    """
    def stringToIntList(self, listOfList):
        result = []
        for listnside in listOfList:
            result.append(list(map(float, listnside)))
        return result
    
    """
        Pre: ---
        Post: Devolvemos un conjunto de variales dependientes y variables dependientes
                para poder entrenar y testear nuestros modelos
    """

    def loadTrainingData(self):
        return self.divideData(self.trainDataFrame), self.divideData(self.testDataFrame)
    
    def divideData(self, data):

        X = data[list(set(data.columns) - set(["Diagnosis"]))].astype(float).values

        Y = data["Diagnosis"].astype(int).values 


        return [X, Y]