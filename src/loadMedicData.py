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
                result = '{0:.3f}'.format(1 + ((value - minValue) / (maxValue - minValue)))
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
            TrainDataPath = "TrainTadpole.csv"
            EvalDataPath = "EvalTadpole.csv"
            if not os.path.exists(TrainDataPath) or not os.path.exists(EvalDataPath):
                # Load D1_D2 evaluation data set
                D1D2Path = Path(medicalPaths[0])
                D1D2df = pd.read_csv(D1D2Path, sep = ",", decimal=".", float_precision='high')              # [12741 rows x 1907 columns]

                D3Path = Path(medicalPaths[1])
                D3df = pd.read_csv(D3Path, sep = ",", decimal=".", float_precision='high')
                        
                D4Path = Path(medicalPaths[2])
                D4df = pd.read_csv(D4Path, sep = ",", decimal=".", float_precision='high')

                print( "Censoring D1D2 dataframe..." )
                self.censor_d1_table(D1D2df)
                print("D1D2 dataframe censored.")
                
            
                # Fill DXCHANGE nans by older values
                D1D2df = self.fill_diagnans_by_older_values( D1D2df )
                # D1D2df = self.fill_nans_by_older_values(D1D2df)
                # Borrando indices que el valor DXCHANGE es NAN  
                print( "Delete dataframe's index with DXCHANGE NaN...")
                idx = D1D2df[np.isnan(D1D2df.DXCHANGE)].index
                D1D2df = D1D2df.drop(idx)
                print( "Delete done.")
                
                D1D2df = D1D2df[features]


                # PREPROCESS STAGE
                if problemType :
                    D1D2df = D1D2df.loc[(D1D2df['DX_bl'] == "LMCI") | ((D1D2df['DX_bl'] == "EMCI") & ((D1D2df['DX'] != "NL") | (D1D2df['DX'] != "MCI to NL")))]
                    D1D2df = self.sMCIpMCIDiagnosisTADPOLE(D1D2df)
                else: 
                    D1D2df = self.obtainDiagnosisTADPOLE(D1D2df)

                if(not usingDX):
                    D1D2df.drop(["DX_bl"], axis=1, inplace=True)
                    D1D2df.drop(["DXCHANGE"], axis=1, inplace=True)
                    D1D2df.drop(["DX"], axis=1, inplace=True)
                
                self.d1d2DataFrame = self.preprocess(D1D2df, featureTypePath)

                # TRAIN AND TEST DATAFRAME NO DATA LEAKAGE USING D3 AS EVAL AND D1-D2 AS TRAIN-TEST
                trainData = self.d1d2DataFrame.copy()
                #testData = self.d1d2DataFrame.copy()
                testData = []
                evalData = self.d1d2DataFrame.copy()

                evalData = evalData[evalData['RID'].isin(D4df['RID'].unique())]
                
                #testData = testData[testData['RID'].isin(D3df['RID'].unique())]
                #testData = testData[~testData['RID'].isin(evalData['RID'])]

                #trainData = trainData[~trainData['RID'].isin(testData['RID'])]
                trainData = trainData[~trainData['RID'].isin(evalData['RID'])]
                
                #testData.drop(["PTID", "RID"], axis=1, inplace=True)
                trainData.drop(["RID"], axis=1, inplace=True)
                evalData.drop(["RID"], axis=1, inplace=True)

                trainData.to_csv(TrainDataPath, index=False, sep = ";", float_format="%.3f")
                evalData.to_csv(EvalDataPath, index=False, sep = ";", float_format="%.3f")

                return [trainData, evalData]

            else:
                trainData = pd.read_csv(TrainDataPath, sep = ";")
                evalData = pd.read_csv(EvalDataPath, sep = ";")

                return [trainData, evalData]


        def loadADNI(self, medicalPaths, features, featureTypePath, problemType, usingDX):
            TrainDataPath = "TrainTadpole.csv"
            EvalDataPath = "EvalTadpole.csv"
            if not os.path.exists(TrainDataPath) or not os.path.exists(EvalDataPath):
                
                ADNIPath = Path(medicalPaths)
                ADNIf = pd.read_csv(ADNIPath, sep = ",", decimal=".", float_precision='high')
                ADNIf = ADNIf[features]

                if problemType :
                    ADNIf = ADNIf.loc[(ADNIf['DX_bl'] == "LMCI") | ((ADNIf['DX_bl'] == "EMCI") & ((ADNIf['DX'] != "NL") | (ADNIf['DX'] != "MCI to NL")))]
                    ADNIf = self.sMCIpMCIDiagnosisTADPOLE(ADNIf)
                else: 
                    ADNIf = self.obtainDiagnosisADNI(ADNIf)

                if(not usingDX):
                    ADNIf.drop(["DX_bl"], axis=1, inplace=True)
                    ADNIf.drop(["DX"], axis=1, inplace=True)


                self.ADNIDataframe = self.preprocess(ADNIf, featureTypePath)
                self.ADNIDataframe.to_csv("test.csv", index=False, sep = ";")
                
                exit()

                trainData.drop(["RID"], axis=1, inplace=True)
                evalData.drop(["RID"], axis=1, inplace=True)

                trainData.to_csv(TrainDataPath, index=False, sep = ";", float_format="%.3f")
                evalData.to_csv(EvalDataPath, index=False, sep = ";", float_format="%.3f")

                return [trainData, evalData]

            else:
                trainData = pd.read_csv(TrainDataPath, sep = ";")
                evalData = pd.read_csv(EvalDataPath, sep = ";")

                return [trainData, evalData]

        def loadClinicData(self, clinicPaths):
            dataPath = Path(clinicPaths[0])
            demo = pd.read_csv(dataPath)    
            
            dataPath = Path(clinicPaths[1])
            neuro = pd.read_csv(dataPath)
            
            dataPath = Path(clinicPaths[2])
            clinical = pd.read_csv(dataPath).rename(columns={"PHASE":"Phase"})

            dataPath = Path(clinicPaths[3])
            #diag = pd.read_csv(dataPath)
            comb = pd.read_csv(dataPath)[["RID", "PTID", "Phase", "DXCURREN", "DXCHANGE", "DIAGNOSIS"]]

            #diag = self.obtainDiagnosis(dataPath, Path(clinicPaths[2]), ).copy()
            #diag = diag.rename(columns = {"Subject": "PTID"})

            clinicalDataFrame = comb.merge(demo, on = ["RID", "Phase"]).merge(neuro, on = ["RID", "Phase"]).merge(clinical, on = ["RID", "Phase"]).drop_duplicates()

            
            clinicalDataFrame.columns = [c[:-2] if str(c).endswith(("_x", "_y")) else c for c in clinicalDataFrame.columns]
            clinicalDataFrame = clinicalDataFrame.loc[:, ~clinicalDataFrame.columns.duplicated()]

            clinicalDataFrame = clinicalDataFrame.replace("-4",-4)
            clinicalDataFrame = clinicalDataFrame.replace(-4, np.NaN)
            
            result = self.preprocess(clinicalDataFrame)
            
            result = self.droppedColumnsDataFrame(result)
            
            print(result.shape)

            return result


        def obtainDiagnosis(self, pathDiag, clinicalPath, combPath):
            
            dataframe = pd.read_csv(pathDiag, index_col="PTID")
            combDF = pd.read_csv(combPath)[["RID", "PTID", "Phase"]]
            clinicalDF = pd.read_csv().rename(columns={"PHASE":"Phase"})

            
            diagnosisDict = {}
            print(dataframe.to_string)
            for key, row in dataframe.iterrows():
                adniPhase = row["Phase"]
                diagnosis = -1
                
                if adniPhase == "ADNI1":
                    diagnosis = row["DXCURREN"]
                elif adniPhase == "ADNI2" or adniPhase == "ADNIGO":

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
                    DXChange = row["DXCHANGE"]
                    diagnosis = row["DXCHANGE"]
                    if DXChange == 7 or DXChange == 9:
                        diagnosis = 1
                    if DXChange == 4 or DXChange == 8:
                        diagnosis = 2
                    if DXChange == 5 or DXChange == 6:
                        diagnosis = 3

                elif adniPhase == "ADNI3":
                    diagnosis = row["DIAGNOSIS"]
                else:   
                    print("NO SE HA RECONOCIDO LA FASE DE ESTUDIO: ", adniPhase)
                    exit()
                if not math.isnan(diagnosis):
                    diagnosisDict[key] = diagnosis
            
        
            diagnosisDataFrame = pd.DataFrame.from_dict(diagnosisDict, orient = "index").reset_index()

            diagnosisDataFrame[0] = diagnosisDataFrame[0].astype(int) - 1
            diagnosisDataFrame = diagnosisDataFrame.rename(columns = {"index": "PTID", 0:"Diagnosis"})    

            combClinical = combDF.merge(clinicalDF, on = ["PTID", "Phase"])

            result = diagnosisDataFrame.merge(combClinical, on = "PTID", how = "outer")

            result = result[["PTID", "Phase", "Diagnosis"]].drop_duplicates()

            return result

        
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

        def obtainDiagnosisADNI(self, dataframe):
            if 'Diagnosis' not in dataframe.columns:
                if 'DX' in dataframe.columns:
                    """We want to transform 'DX' (a change in diagnosis, in contrast
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

                    dataframe['Diagnosis'] = dataframe['DX']
                    dataframe = dataframe.replace({'Diagnosis': {'CN': 1, 'MCI': 2, 'Dementia': 3}})

            dataframe["Diagnosis"] = dataframe["Diagnosis"] - 1
            return dataframe


        def sMCIpMCIDiagnosisTADPOLE(self, dataframe): 
            PTIDList = dataframe["PTID"].unique()
            resultDataframe = pd.DataFrame(columns = ["PTID", "Diagnosis"])

            for patientID in PTIDList:
                patientDataframe = dataframe.loc[dataframe['PTID'] == patientID][["PTID", "DXCHANGE"]]

                patientDataframe['Diagnosis'] = np.where(any(patientDataframe.loc[((patientDataframe.DXCHANGE == 5) | (patientDataframe.DXCHANGE == 3)), 'PTID']), 1, 0)
                resultDataframe = pd.concat([resultDataframe, patientDataframe[['PTID', 'Diagnosis']]])
            result = pd.merge(dataframe, resultDataframe, left_index=True, right_index=True)
            result.drop(["PTID_y"], axis=1, inplace=True)
            result.rename({'PTID_x': 'PTID'}, axis=1, inplace=True)
            return result
       
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
            
            dataTypeDf = pd.read_csv(Path ("Feature_Type.csv"), sep=';')
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
                            # Pasamos de binario a decimal entre 1 y 2
                            oneHot = oneHot.replace(1,2)
                            oneHot = oneHot.replace(0,1)
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

            result = pd.concat([df[["Diagnosis"]].reset_index(), dfCat.reset_index(), dfQuant.reset_index()], axis=1).drop("index", axis=1) #tex

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