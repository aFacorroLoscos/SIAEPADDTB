
import os
import pandas as pd 
import numpy as np 
from sklearn.impute import KNNImputer
from itertools import repeat 
from contextlib import redirect_stdout 
from pathlib import Path

class Datasets:
    """Resumen de la clase
    Esta clase se utiliza para la carga y procesado del conjunto de datos TADPOLE CHALLENGE
    

    Atributos:
        columns: Lista de nombre de columnas que se utilizan en parte del procesado de los datos
        excludedColumns: Lista de nombre de columnas que no se modifica en el procesado de los datos
        d1d2DataFrame: Dataframe D1 y D2
        ADNIDataframe: Dataframe procedente de ADNI
    """

    def __init__( self ):
        """
        Inicializacion de los parametros utilizados en la clase
        """
        self.columns = ['RID', 'AGE']

        self.excludedColumns = ["RID", "PTID", "Diagnosis", "DXCHANGE"]

        self.d1d2_data_frame = None

    def print_list_into_file(self, value_to_print, filename):
        """
        Funcion que escribe los valores de value_to_print en filename

        :param value_to_print: Valor o lista de valores
        :param filename: Archivo destino
        """
        with open(filename + '.txt', 'w') as f:
            with redirect_stdout(f):
                print('\n'.join(map(str, value_to_print)))


    def normalized_values(self, value, maxValue, minValue):
        """
        Normalizacion del valor value entre maxValue y minValue

        :param value: Valor entero o decimal
        :param maxValue: Valor mayor o igual que el valor value
        :param minValue: Valor menor o igual que el valor value
        :return: Devuelve un valor result que es el valor normalizado de value entre maxValue y minValue
        """
        result = np.nan
        if not pd.isna(value):
            result = '{0:.3f}'.format((value - minValue) / (maxValue - minValue))
        return result

    def standardized_values(self, value, mean, std):
        """
        Estandarizacion del valor value dado un valor de media mean y una desviacion estandar std

        :param value: Valor entero o decimal
        :param mean: Media del conjunto de valores
        :param std: Desviacion estandar del conjunto de valores
        :return: Devuelve un valor result que es el valor normalizado de value entre maxValue y minValue
        """
        result = 0
        if not pd.isna(value):
            result = '{0:.3f}'.format((value - mean) / std)
        return result

    def censor_d1_table(self, _table):
        """
        _table pasa a ser un dataframe de datos que no tengan las filas de aquellos valores
        de la columna RID y VISCODE que queremos eliminar.

        :param _table: Dataframe de datos        
        """
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

    def fill_diagnans_by_older_values(self, train_df):
        """
        Cambiamos los valores NANS del dataframe con valores antiguos

        :param train_df: Dataframe de datos con valores nans
        :return: Dataframe df con valores nans cambiados por valores antiguos
        """
        """Fill nans in Diagnosis column from feature matrix by older values (ffill), then by newer (bfill)"""
        df = train_df.copy()
        
        # Ordenacion EMC-EB
        df['AUX'] = df['AGE']
        df['AUX'] += df['Month_bl'] / 12. # Cuidado no este sumando dos veces
        df = df.sort_values(['RID','AUX'])
        
        # Ordenacion mhg
        #train_df = train_df.sort_values(['RID','VISCODE']) # Short values por VISCODE puede fallar en casos como m108
        df_filled_nans = df.groupby('RID')['DXCHANGE'].fillna(method='ffill')
        df['DXCHANGE'] = df_filled_nans
    
        df_filled_nans = df.groupby('RID')['DXCHANGE'].fillna(method='bfill')
        df['DXCHANGE'] = df_filled_nans

        df_filled_nans = df.groupby('RID')['DX'].fillna(method='ffill')
        df['DX'] = df_filled_nans
    
        df_filled_nans = df.groupby('RID')['DX'].fillna(method='bfill')
        df['DX'] = df_filled_nans
        
        df = df.drop(['AUX'], axis=1)
        return df

    def load_TADPOLE(self, medical_paths, features, problem_type, using_DX):
        """
        Carga de los datos de D1-D2 y D4 a dataframes correspondientes
        Dependiendo del tipo de problema, el diagnostico de los pacientes pasa de un diagnostico
        CN/MCI/AD a un diagnostico sMCI/pMCI teniendo en cuenta solo aquellos pacientes con MCI
        
        :param medical_paths: Paths de archivos origen de los datos medicos de TADPOLE CHALLENGE
        :param features: Lista de features a tener en cuenta en el dataframe
        :param problem_type: Bool, TRUE si es problema sMCI/pMCI, False es problema CN/MCI/AD
        :using_DX: Bool, False si no se quiere utilizar valores de diagnosis, True en caso contrario
        """
        train_data_path = "../TrainTadpole.csv"
        eval_data_path = "../EvalTadpole.csv"
        if not os.path.exists(train_data_path) or not os.path.exists(eval_data_path):
            # Load D1_D2 evaluation data set
            D1D2_path = Path(medical_paths[0])
            D1D2_df = pd.read_csv(D1D2_path, sep = ",", decimal=".", float_precision='high')              # [12741 rows x 1907 columns]

            D4_path = Path(medical_paths[2])
            D4_df = pd.read_csv(D4_path, sep = ",", decimal=".", float_precision='high')

            print( "Censoring D1D2 dataframe..." )
            self.censor_d1_table(D1D2_df)
            print("D1D2 dataframe censored.")
        
            # Fill DXCHANGE nans by older values
            D1D2_df = self.fill_diagnans_by_older_values( D1D2_df )
            # Borrando indices que el valor DXCHANGE es NAN  
            print( "Delete dataframe's index with DXCHANGE NaN...")
            idx = D1D2_df[np.isnan(D1D2_df.DXCHANGE)].index
            D1D2_df = D1D2_df.drop(idx)
            print( "Delete done.")
            
            D1D2_df = D1D2_df[features]

            # PREPROCESS STAGE
            if problem_type :
                D1D2_df = D1D2_df.loc[((D1D2_df['DX_bl'] == "LMCI") | (D1D2_df['DX_bl'] == "EMCI")) & (D1D2_df['DX'] != "NL") & (D1D2_df['DX'] != "MCI to NL")]
                D1D2_df = self.sMCI_pMCI_diagnosis_TADPOLE(D1D2_df)
            else: 
                D1D2_df = self.obtain_diagnosis_TADPOLE(D1D2_df)

            if(not using_DX):
                D1D2_df.drop(["DX_bl"], axis=1, inplace=True)
                D1D2_df.drop(["DXCHANGE"], axis=1, inplace=True)
                D1D2_df.drop(["DX"], axis=1, inplace=True)
            D1D2_df.drop(["EXAMDATE"], axis=1, inplace=True)
            
            self.d1d2_data_frame = self.preprocess(D1D2_df)

            # TRAIN AND TEST DATAFRAME NO DATA LEAKAGE USING D3 AS EVAL AND D1-D2 AS TRAIN-TEST
            train_data = self.d1d2_data_frame.copy()
            eval_data = self.d1d2_data_frame.copy()
            eval_data = eval_data[eval_data['RID'].isin(D4_df['RID'].unique())]
            train_data = train_data[~train_data['RID'].isin(eval_data['RID'])]
            
            train_data.drop(["RID"], axis=1, inplace=True)
            eval_data.drop(["RID"], axis=1, inplace=True)

            train_data.to_csv(train_data_path, index=False, sep = ";", float_format="%.3f")
            eval_data.to_csv(eval_data_path, index=False, sep = ";", float_format="%.3f")
            
            return [train_data, eval_data]

        else:
            train_data = pd.read_csv(train_data_path, sep = ";")
            eval_data = pd.read_csv(eval_data_path, sep = ";")

            return [train_data, eval_data]

    
    def obtain_diagnosis_TADPOLE(self, dataframe):
        """
        Se crea una columna nueva llamada Diagnosis en caso de no existir,
        la columna diagnosis se obtiene a partir de la columna DXCHANGE, donde
        los valores son diagnosticos que cambian de una visita a otra, se cambian
        los valores por diagnosticos estables entre visitas

        :param dataframe: Dataframe de datos
        :return: El dataframe dataframe tiene una nueva columna llamada diagnosis
        donde se cambia el valor DXCHANGE por un valor estable de diagnostico, en caso de
        que no exista previamente
        """
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

    def sMCI_pMCI_diagnosis_TADPOLE(self, dataframe): 
        """
        Dividimos los pacientes diagnosticados con MCI entre pacientes con un diagnostico sMCI
        y un diagnostico pMCI, dependiendo de si el paciente padece AD en un plazo de 3 años
        entre consultas

        :param dataframe: Dataframe de datos
        :return: Se devuelve un dataframe con una columna llamada Diagnosis donde el valor es 0
        si el paciente padece de sMCI, 1 en caso de padecer pMCI
        """
        PTID_list = dataframe["PTID"].unique()

        result_dataframe = pd.DataFrame()

        for patient_ID in PTID_list:
            # Nos quedamos con solo los pacientes con el mismo PTID
            patient_diagnosis = dataframe.loc[dataframe['PTID'] == patient_ID].copy()

            # Solo tenemos en cuenta para el nuevo diagnostico sMCI o pMCI aquellos con un plazo de 3 años entre consultas
            patient_diagnosis['EXAMDATE'] = pd.to_datetime(patient_diagnosis["EXAMDATE"], format='%Y-%m-%d', errors='coerce')
            minimun_year = patient_diagnosis['EXAMDATE'].min() 
            

            patient_3_years = patient_diagnosis.loc[patient_diagnosis["EXAMDATE"] <= minimun_year + np.timedelta64(3,'Y')].copy()
            
            # Si se cuentra algún paciente con DEMENTIA or MCI to DEMENTIA entonces es Diagnostic = 1, 0 en caso contrario
            if len(patient_3_years.loc[((patient_3_years.DXCHANGE == 5) | (patient_3_years.DXCHANGE == 3))]) == 0:
                patient_3_years["Diagnosis"] = 0
            else:
                patient_3_years["Diagnosis"] = 1

            # Lo guardamos en un dataframe
            result_dataframe = pd.concat([result_dataframe, patient_3_years])

        return result_dataframe
    
    """
        Pre: ---
        Post: Realiza el procesamiento del datafremo
    """
    def preprocess(self, dataframe):
        """
        Etapa de procesamiento de los datos, los datos cuantitativos se convierten en
        valores entre 0 y 1, normalizandolos con los valores correspondientes a sus columnas
        Los valores cualitativos se realiza un proceso One Hot Encoding. Las columnas con mas del
        70% de valores NAN, vacios o un valor -4 (equivalente a NAN) se eliminan automaticamente,
        al igual que las columnas que no nos dan datos importantes, como las columnas de fechas
        ids...etc

        Los valores NAN restantes se rellenan mediante KNNImputer

        :param dataframe: Dataframe de datos
        :return: Un dataframe tras ejecutar la etapa de procesado
        """
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
        
        df_quant = pd.DataFrame()
        df_cat = pd.DataFrame()
        
        data_type_Df = pd.read_csv(Path ("../Feature_Type.csv"), sep=';')

        for (column_name, column_data) in df.items():
            # Columnas que no queremos tener en cuenta
            if(column_name.strip() not in self.excludedColumns):
                data_row = data_type_Df.loc[data_type_Df["COLUMN"] == column_name]["TYPE"]

                # Columnas de tipo string
                if(column_name in data_type_Df["COLUMN"].values and data_row.values[0] == "T" ):
                    if(column_data.isnull().mean() < 0.70 or column_name == "PTETHCAT"):
                        # Rellenamos la columna con datos por defecto
                        column_data = column_data.fillna(value='None')
                        # Realizamos el proceso de one hot encoding
                        one_hot = pd.get_dummies(column_data, column_name, drop_first=True, dtype=int)
                        if(df[column_name].isnull().values.any()):
                            one_hot = one_hot.drop([column_name + "_None"], axis = 1)
                        df_cat = pd.concat([df_cat, one_hot], axis = 1)
                    
                # Caso en que las columnas sean numericas
                elif(column_name in data_type_Df["COLUMN"].values and data_row.values[0] == "N" ):
                    if(column_data.isnull().mean() < 0.70):
                        column_data = pd.to_numeric(column_data)
                        # Convertimos todos los valores a valores numeros entre 1 y 2
                        minValue = np.nanmin(column_data)
                        maxValue = np.nanmax(column_data)
                        if(minValue < 0):
                                absValue = abs(minValue)
                                column_data = column_data + absValue 
                                minValue = np.nanmin(column_data)
                                maxValue = np.nanmax(column_data)
                        column_data = list(map(self.normalized_values, column_data, repeat(maxValue), repeat(minValue)))
                        # mean = column_data.mean()
                        # std = column_data.std()
                        #column_data = list(map(self.standardized_values, column_data, repeat(mean), repeat(std)))
                        newdf = pd.DataFrame(columns = [column_name])
                        newdf[column_name] = column_data
                        df_quant = pd.concat([df_quant, newdf], axis=1)
                # Caso en el que las columnas sean de un tipo desconocido
                else:
                    df.drop(column_name, axis=1, inplace=True)    

        imputer = KNNImputer(n_neighbors=10)
        result = pd.concat([df[["Diagnosis"]].reset_index(), df_cat.reset_index(), df_quant.reset_index()], axis=1).drop("index", axis=1)
        result = pd.DataFrame(imputer.fit_transform(result),columns = result.columns)
        result = pd.concat([df[["RID"]].reset_index(), result.reset_index()], axis=1).drop("index", axis=1) #tex

        print("Pre-processing finished!")
        return result

    
    def divide_data(self, data):

        X = data[list(set(data.columns) - set(["Diagnosis"]))].astype(float).values
        Y = data["Diagnosis"].astype(int).values 


        return [X, Y]