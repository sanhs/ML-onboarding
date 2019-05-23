#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

BASE_DIR = 'E:\\workspace\\python\scikitlearn\\onbaording2\\ML-onboarding\\Patient Readmission Assignment\\'
patient_data = pd.read_csv(BASE_DIR + 'Patientdata.csv')
hospital_data = pd.read_csv(BASE_DIR + 'HospitalData.csv')
diagnosis_data = pd.read_csv(BASE_DIR + 'DiagnosisData.csv')

diagnosis_data['num_medications'] = pd.cut(diagnosis_data['num_medications'], bins=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])

# seperate test data out
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

# raw_data
data = patient_data.merge(hospital_data.merge(diagnosis_data, on='patientID'), on='patientID')

data = data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications', 'Target']]

# Label Encoding
le = LabelEncoder()
ordinal_cols = ['age', 'Target', 'change', 'diabetesMed']
for col in ordinal_cols:
    data[col] = le.fit_transform(data[col].astype(str))

X = data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']]
y = data['Target']

from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=250)
model.fit(X, y)

def get_test_data_output():
    final_test = test_data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']]
    for col in ['age', 'change', 'diabetesMed']:
        final_test[col] = le.fit_transform(final_test[col].astype(str))
    test_data_predictions = model.predict(final_test)
    
    final_output['patientID'] = test_data['patientID'].astype(str)
    final_output['Target'] = test_data_predictions
    final_output = final_output[['patientID', 'Target']]
    
    file_name = 'E:\\workspace\\python\\scikitlearn\\onbaording2\\ML-onboarding\\Patient Readmission Assignment\\final_output.csv'
    final_output.to_csv(file_name, sep='\t')


def get_prediction(patient_data):
    return 'No'



# create a server in nodesjs
# create api to send patient data to server
# integrate server with python 
# send data from server to python 
# convert data to format model can take -- don't know how
# get prediction
# send it back to server