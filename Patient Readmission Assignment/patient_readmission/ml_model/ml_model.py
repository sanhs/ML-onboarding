def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import make_pipeline

print('loading data...')
BASE_DIR = 'E:\\workspace\\python\scikitlearn\\onbaording2\\ML-onboarding\\Patient Readmission Assignment\\'
patient_data = pd.read_csv(BASE_DIR + 'Patientdata.csv')
hospital_data = pd.read_csv(BASE_DIR + 'HospitalData.csv')
diagnosis_data = pd.read_csv(BASE_DIR + 'DiagnosisData.csv')

print('data loaded...')
# seperating test data out
# patient
patient_test_data = patient_data[patient_data['istrain'] == 0]
patient_data = patient_data[patient_data['istrain'] == 1]
# hospital
hospital_test_data = hospital_data[hospital_data['istrain'] == 0]
hospital_data = hospital_data[hospital_data['istrain'] == 1]
# diagnosis
diagnosis_test_data = diagnosis_data[diagnosis_data['istrain'] == 0]
diagnosis_data = diagnosis_data[diagnosis_data['istrain'] == 1]

# test_data
test_data = patient_test_data.merge(hospital_test_data.merge(diagnosis_test_data, on='patientID'), on='patientID')
X_test = test_data[test_data.columns.difference(['Target'])]
y_test = test_data['Target']

# raw_data
train_data = patient_data.merge(hospital_data.merge(diagnosis_data, on='patientID'), on='patientID')
X_train = train_data[train_data.columns.difference(['Target'])]
y_train = train_data['Target']

class PreProcessing(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def transform(self, df):
        pred_var = ['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']
 
        df = df[pred_var]
        # TODO: find out how to deal with bins later
        # diagnosis_data['num_medications'] = pd.cut(diagnosis_data['num_medications'], bins=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])

        age_values = {'[90-100)':9, '[50-60)':5, '[80-90)':8, '[60-70)':6, '[70-80)':7, '[30-40)':3, 
            '[40-50)':4, '[20-30)':2, '[10-20)':1, '[0-10)':0}
        change_values = {'Ch': 1, 'No': 1}
        diabetesMed_values = {'Yes': 1, 'No': 0}
        # Target_values = {'Yes': 1, 'No': 0}

        df = df.replace({
            'age': age_values,
            'change': change_values,
            'diabetesMed': diabetesMed_values,
            # 'Target': Target_values
        })
        print('data processing complete...')
        return df.as_matrix()
    
    def fit(self, df, y=None, **fit_params):
        print('processing data...')
        return self


y_train = y_train.replace({'Y': 1, 'N': 0}).as_matrix()
y_test = y_test.replace({'Y': 1, 'N': 0}).as_matrix()

data_processor = PreProcessing()
abc = AdaBoostClassifier(n_estimators=250)
model = make_pipeline(data_processor, abc)
model.fit(X_train, y_train)


import pickle
def save_model():
    with open('model_asd', 'wb') as f:
        pickle.dump(obj=model, file=f)


def load_model():
    with open('model_asd', 'rb') as f:
        return pickle.load(f)



