# Bibliotecas Usadas
import sys
import os
import logging
import math
from types import NoneType
import pandas as pd # Usada para dataflow
import re # Expresiones regulares
import numpy as np # Acceder a datos np.int o np.float
from sklearn.impute import KNNImputer # Rellenar datos NaN del Dataframe
from itertools import repeat # Funcion repeat
from contextlib import redirect_stdout # Redirigir salida a fichero
from pathlib import Path # Manejo de paths

# clase encargada de cargar y procesar los datos de entrada
class Datasets:

    """
        Pre: ---
        Post: Inicializamos la clase Dataset
    """
    def __init__( self ):
        self.columns = ['RID', 'AGE']

        self.excluded_columns = ["RID", "PTID", "Diagnosis", "DXCHANGE"]
        self.d1d2_processed = None

    """
        Pre: maxValue y minValue deben ser mayores que 0
        Post: Devuelve un valor que se encuentra entre 1 y 2
    """
    def normalized_values(self, value, max_value, min_value):
        result = np.nan
        if not pd.isna(value):
            result = '{0:.3f}'.format(1 + ((value - min_value) / (max_value - min_value)))
        return result

    """
        Pre: ---
        Post: Standariza el valor usando la media y varianza
    """
    def standarized_values(self, value, mean, std):
        result = np.nan
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
        _table.drop(_table.loc[(_table.RID == 5204)].index, inplace=True)

        # PLOS One data
        _table.drop(_table.loc[(_table.RID == 2210) & (_table.VISCODE == 'm60')].index, inplace=True)


        _table.reset_index()    

        # --------------------------------------------------------------------------------------

    """
        Pre: train_df es un dataframe
        Post: Devolvemos un dataframe donde se han rellenado aquellos huecos vacios de la columna DX y DXCHANGE
    """
    def fill_diagnans_by_older_values(self, train_df):
        """Fill nans in Diagnosis column from feature matrix by older values (ffill), then by newer (bfill)"""
        
        df = train_df.copy()
        
        # Ordenacion EMC-EB de los datos
        train_df['AUX'] = train_df['AGE']
        train_df['AUX'] += train_df['Month_bl'] / 12. # Cuidado no este sumando dos veces
        train_df = train_df.sort_values(['RID','AUX'])
        
        # Rellenamos los NaNs con valores antiguos (ffill) y despues con valores nuevos (bfill)
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
        Pre: medical_paths contiene la lista de paths hacia los documentos TADPOLE D1,D2,D3 y D4, respectivamente.
             feature_type_path contiene el path del tipo de datos de los features de la lista features
             problem_type y usin_DX deben ser valores entre 0 y 1
        Post: Se devuelven dos conjuntos de datos procesados, train_data y eval_Data para entrenar y validar un modelo
    """
    def load_TADPOLE(self, medical_paths, features, problem_type, using_DX):
        train_data_path = "../TrainTadpole.csv"
        eval_data_path = "../EvalTadpole.csv"
        # Para no sustituir por accidente los dos conjuntos de datos en caso de que existan 
        # devuelve el conjunto de datos ya especificado en el path
        if not os.path.exists(train_data_path) or not os.path.exists(eval_data_path):
            # Cargamos los dataframes D1,D2,D3 y D4 de TADPOLE
            d1d2_path = Path(medical_paths[0])
            d1d2_dataframe = pd.read_csv(d1d2_path, sep = ",", decimal=".", float_precision='high')              # [12741 rows x 1907 columns]

            d3_path = Path(medical_paths[1])
            d3_dataframe = pd.read_csv(d3_path, sep = ",", decimal=".", float_precision='high')
                    
            d4_path = Path(medical_paths[2])
            d4_dataframe = pd.read_csv(d4_path, sep = ",", decimal=".", float_precision='high')

            print( "Censoring dataframe stage..." )
            self.censor_d1_table(d1d2_dataframe)
        
            print( " Fill NAN by olders and newers values...")
            d1d2_dataframe = self.fill_diagnans_by_older_values( d1d2_dataframe ) 

            print( "Delete dataframe's index with DXCHANGE NaN...")
            idx = d1d2_dataframe[np.isnan(d1d2_dataframe.DXCHANGE)].index
            d1d2_dataframe = d1d2_dataframe.drop(idx)
            
            print ( "Processing values stage...")
            d1d2_dataframe = d1d2_dataframe[features]
            # PPROCESS STAGE
            # Dependiendo del tipo de problema, CN-MCI-AD o sMCI-pMCI tendremos distintos tipos de diagnosticos
            if problem_type :
                d1d2_dataframe = d1d2_dataframe.loc[(d1d2_dataframe['DX_bl'] == "LMCI") | ((d1d2_dataframe['DX_bl'] == "EMCI") & ((d1d2_dataframe['DX'] != "NL") | (d1d2_dataframe['DX'] != "MCI to NL")))]
                d1d2_dataframe = self.sMCI_pMCI_diagnosis_TADPOLE(d1d2_dataframe)
            else: 
                d1d2_dataframe = self.diagnosis_TADPOLE(d1d2_dataframe)

            if(not using_DX):
                d1d2_dataframe.drop(["DX_bl"], axis=1, inplace=True)
                d1d2_dataframe.drop(["DXCHANGE"], axis=1, inplace=True)
                d1d2_dataframe.drop(["DX"], axis=1, inplace=True)
            
            d1d2_dataframe.drop(["EXAMDATE"], axis=1, inplace=True)
            
            self.d1d2_processed = self.process_dataframe(d1d2_dataframe)

            train_data = self.d1d2_processed.copy()
            eval_data = self.d1d2_processed.copy()

            # Creamos el conjunto de datos de evaluacion a partir del dataframde D4 Tapole
            # Eliminamos los datos D4 que estan dentro de D1-D2, asi no hay DATALEAKAGE
            eval_data = eval_data[eval_data['RID'].isin(d4_dataframe['RID'].unique())]
            train_data = train_data[~train_data['RID'].isin(eval_data['RID'])]
            
            train_data.drop(["RID"], axis=1, inplace=True)
            eval_data.drop(["RID"], axis=1, inplace=True)

            # Creamos los CSV con los datos preprocesados
            train_data.to_csv(train_data_path, index=False, sep = ";", float_format="%.3f")
            eval_data.to_csv(eval_data_path, index=False, sep = ";", float_format="%.3f")
            
            return [train_data, eval_data]

        else:
            train_data = pd.read_csv(train_data_path, sep = ";")
            eval_data = pd.read_csv(eval_data_path, sep = ";")

            return [train_data, eval_data]

    """
        Pre: ---
        Post: Devuelve un dataframe con una columna añadida en caso de no existir llamada Diagnosis que contienen
              valores entre 0 y 2 según el tipo de clase de diagnostico que tenga el paciente
    """
    def diagnosis_TADPOLE(self, dataframe):
        if 'Diagnosis' not in dataframe.columns:
            if 'DXCHANGE' in dataframe.columns:
            
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
                # valores DXCHANGE segun lo estipulado anteriormente a un diagnostico actual
                # la columna DXCHANGE contiene cambios en el diagnosticos en contraste con la anterior visita

                dataframe['Diagnosis'] = dataframe['DXCHANGE']
                dataframe = dataframe.replace({'Diagnosis': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})

        # Les restamos un valor para que se encuentren en 0 y 2, no influye en los resultados
        dataframe["Diagnosis"] = dataframe["Diagnosis"] - 1
        return dataframe

    """
        Pre: ---
        Post: Modificamos la columna Diagnosis en caso de no existir para que contenga pacientes que han sido 
              diagnosticados con sMCI o pMCI, por lo que la columna Diagnosis tendrá valores entre 0 y 1
    """
    def sMCI_pMCI_diagnosis_TADPOLE(self, dataframe): 
        if 'Diagnosis' not in dataframe.columns:
            if 'DXCHANGE' in dataframe.columns:
                patient_list = dataframe["PTID"].unique()

                result_dataframe = pd.DataFrame(columns = ["PTID", "Diagnosis"])
                for patient in patient_list:
                    # Creamos un dataframe de todos los diagnosticos del paciente
                    patient_dataframe = dataframe.loc[dataframe['PTID'] == patient][["PTID", "DXCHANGE", "EXAMDATE"]]

                    # Solo tenemos en cuenta para el nuevo diagnostico sMCI o pMCI aquellos con un plazo de 3 años entre consultas
                    patient_dataframe["EXAMDATE"] = pd.to_datetime(patient_dataframe["EXAMDATE"], format='%Y-%m-%d', errors='coerce')
                    minimun_year = min(patient_dataframe['EXAMDATE'])
                    patient_3_years = patient_dataframe.loc[patient_dataframe["EXAMDATE"] <= minimun_year + np.timedelta64(3,'Y')]

                    # Si se cuentra algún paciente con DEMENTIA or MCI to DEMENTIA en la columna DXCHANGE
                    # entonces es Diagnostic = 1, 0 en caso contrario
                    if len(patient_3_years.loc[((patient_3_years.DXCHANGE == 5) | (patient_3_years.DXCHANGE == 3))]) == 0:
                        patient_dataframe["Diagnosis"] = 0
                    else:
                        patient_dataframe["Diagnosis"] = 1

                    resultDataframe = pd.concat([resultDataframe, patient_dataframe[['PTID', 'Diagnosis']]])
                
                # Hacemos merge de todos los PTDI de los pacientes con la columna Diagnostic creada
                result_dataframe = pd.merge(dataframe, result_dataframe, left_index=True, right_index=True)
                result_dataframe.drop(["PTID_y"], axis=1, inplace=True)
                result_dataframe.rename({'PTID_x': 'PTID'}, axis=1, inplace=True)

                return result_dataframe
    
    """
        Pre: featurePath debe apuntar al fichero que contiene el tipo de datos de los features
        Post: Devuelve un dataframe con valores que se encuentran entre 1 y 2
    """
    def preprocess(self, dataframe, feature_path):

        print("Pre-processing...")

        df = dataframe.copy()

        # Añadimos meses a los años
        if 'Month_bl' in df.columns:
            df['AGE'] += df['Month_bl'] / 12.
            df = df.drop(["Month_bl"], axis=1)
        

        # Convertimos la tabla DX y DX_bl que contienen string a numeros
        # Para facilitar el one hot encoding
        if 'DX_bl' in df.columns and type(df['DX_bl'].iloc[0]) is str:
            df = df.replace({'DX_bl': {'CN': 1, 'AD': 3, 'EMCI': 2, 'LMCI': 2, 'SMC': 2 }})
        if 'DX' in df.columns and type(df['DX'].iloc[0]) is str:   
            df = df.replace({'DX': {'Dementia': 3, 'Dementia to MCI': 8, 'MCI': 2, 'MCI to Dementia': 5, 'MCI to NL': 7, 'NL': 1, 'NL to MCI': 4 }})
            df = df.replace({'DX': {4: 2, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1}})    

        # Reemplazamos los valores vacios por NAN, los -4 tambien son valores NAN
        df = df.replace(' ', np.nan, regex=True)    
        df = df.replace('', np.nan, regex=True)
        df = df.replace("-4", -4, regex=True)
        df = df.replace(-4, np.nan, regex=True)
        
        df_quant = pd.DataFrame()
        df_cat = pd.DataFrame()
        
        data_type_df = pd.read_csv(Path ("../Feature_Type.csv"), sep=';')

        for (column_name, column_data) in df.items():
            # Columnas que no queremos tener en cuenta
            if(column_name.strip() not in self.excludedColumns):
                data_row = data_type_df.loc[data_type_df["COLUMN"] == column_name]["TYPE"]

                # Columnas de tipo string
                if(column_name in data_type_df["COLUMN"].values and data_row.values[0] == "T" ):
                    
                    # En caso de que más del 70% sean datos NULL entonces no se tiene en cuenta
                    if(column_data.isnull().mean() < 0.70 or data_type_df == "PTETHCAT"):

                        column_data = column_data.fillna(value='None')

                        # Proceso de one hot encoding
                        one_hot = pd.get_dummies(column_data, column_data, drop_first=True, dtype=int)

                        one_hot = one_hot.replace(1,2)
                        one_hot = one_hot.replace(0,1)
                        
                        if(df[column_name].isnull().values.any()):
                            one_hot = one_hot.drop([column_name + "_None"], axis = 1)

                        df_cat = pd.concat([df_cat, one_hot], axis = 1)
                        

                # Caso en que las columnas sean numericas
                elif(column_name in data_type_df["COLUMN"].values and data_row.values[0] == "N" ):

                    if(column_data.isnull().mean() < 0.70):
                        column_data = pd.to_numeric(column_data)
                        max_value = np.nanmax(column_data)
                        min_value = np.nanmin(column_data)

                        # Si hay algún valor negativo
                        if(min_value < 0):
                                abs_value = abs(min_value)
                                column_data = column_data + abs_value 
                                min_value = np.nanmin(column_data)
                                max_value = np.nanmax(column_data)

                        # Normalizamos todos los datos entre valores numericos que están entre 1 y 2
                        column_data = list(map(self.normalized_values, column_data, repeat(max_value), repeat(min_value)))

                        new_df = pd.DataFrame(columns = [column_name])
                        new_df[column_name] = column_data
                        df_quant = pd.concat([df_quant, new_df], axis=1)

                # Caso en el que las columnas sean de un tipo desconocido
                else:
                    df.drop(column_name, axis=1, inplace=True)    

        result = pd.concat([df[["Diagnosis"]].reset_index(), df_cat.reset_index(), df_quant.reset_index()], axis=1).drop("index", axis=1) #tex

        # Rellenamos los atributos NAN con valores vecinos al susodicho utilizando KNNImputer
        imputer = KNNImputer(n_neighbors=10)
        result = pd.DataFrame(imputer.fit_transform(result),columns = result.columns)
        result = pd.concat([df[["RID"]].reset_index(), result.reset_index()], axis=1).drop("index", axis=1)

        print("Pre-processing finished!")

        return result