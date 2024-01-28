import os 
import pickle
import time
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def colPreparation():
    labelEncoder = ['Gender','Driving_License','Previously_Insured','Vehicle_Damage']
    oneHotEncoder = ['Vehicle_Age','Region_Code','Policy_Sales_Channel']
    scallingStandar = ['Age','Annual_Premium','Vintage']
    
    return labelEncoder, oneHotEncoder, scallingStandar

class InsuranceModel():
    def __init__(self) -> None:
        pass

    def runModel(self, data, typed='multi'):
        path = os.getcwd() + '/modules/packages/'
        model = pickle.load(open(path + 'model_InsuranceRecommendation.pkl', 'rb'))
        col_p = pickle.load(open(path + 'columnPreparation.pkl', 'rb'))
        col_m = pickle.load(open(path + 'columnModelling.pkl', 'rb'))

        X = data[col_p]
        colEncoder, colpOneHotEncoder, colStandarScaler = colPreparation()
        for col in X.columns:
            prep = pickle.load(open(path + 'prep' + col + '.pkl', 'rb'))
            if col in colpOneHotEncoder:
                dfTemp = pd.DataFrame(prep.transform(X[[col]]).toarray())
                X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
            else:
                dfTemp = pd.DataFrame(prep.transform(X[[col]]))
                X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
        X.columns = col_m
    
        if typed == 'multi':
            y = model.predict(X)
            return y
    
        elif typed == 'single':
            y = model.predict(X)[0]
            if y == 0:
                return 0
            else:
                return 1
        else:
            return False