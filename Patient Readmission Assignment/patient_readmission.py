#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# # Import Data

# In[162]:


BASE_DIR = 'E:\\workspace\\python\scikitlearn\\onbaording2\\ML-onboarding\\Patient Readmission Assignment\\'
patient_data = pd.read_csv(BASE_DIR + 'Patientdata.csv')
print(patient_data.shape)
patient_data.head()


# In[163]:


sns.countplot(patient_data['race'])


# In[164]:


sns.countplot(patient_data.loc[patient_data['Target'] == 'Yes', 'gender'])


# In[165]:


sns.countplot(patient_data.loc[patient_data['Target'] == 'Yes', 'race'])


# In[166]:


sns.countplot(patient_data.loc[patient_data['Target'] == 'No', 'race'])


# In[167]:


hospital_data = pd.read_csv(BASE_DIR + 'HospitalData.csv')
print(hospital_data.shape)
hospital_data.head()


# In[168]:


diagnosis_data = pd.read_csv(BASE_DIR + 'DiagnosisData.csv')
print(diagnosis_data.shape)
diagnosis_data.head()


# # Data Cleaning

# In[169]:


patient_data.isnull().sum()


# __Too many missing values, Dropping weight column__

# In[170]:


patient_data = patient_data.drop(columns=['weight'])
patient_data.isnull().sum()


# In[171]:


patient_data.groupby(['gender', 'Target']).count()


# In[172]:


patient_data['Target'].value_counts()


# In[173]:


hospital_data.isnull().sum()


# In[174]:


hospital_data.groupby(['payer_code', 'medical_specialty']).count().shape


# In[175]:


for col in hospital_data.columns.values:
    print(col + ' - ms', hospital_data.groupby([col, 'medical_specialty']).count().shape)
    print(col + ' - pc', hospital_data.groupby([col, 'payer_code']).count().shape)
    print(col + ' - pc + ms', hospital_data.groupby([col, 'payer_code', 'medical_specialty']).count().shape)


# In[176]:


hospital_data['admission_type_id'].value_counts()


# In[177]:


for i in range(1, 9):
    temp_1 = hospital_data[hospital_data['admission_type_id'] == i].isnull().sum()
    print(i, temp_1['payer_code'], temp_1['medical_specialty'])


# In[178]:


hospital_data.isnull().sum()[['payer_code', 'medical_specialty']]


# __Could not find any correlation between payer_code and medical_specialty and other columns__
# 
# since missing values are close to 50% dropping both the columns

# In[179]:


hospital_data = hospital_data.drop(columns=['payer_code', 'medical_specialty'])
hospital_data.shape


# In[180]:


hospital_data[hospital_data['admission_type_id'] == 4]


# In[181]:


hospital_data['Admission_date'] = pd.to_datetime(hospital_data['Admission_date'], format='%Y-%m-%d')
hospital_data['Admission_date'].dtype


# In[182]:


hospital_data['Admission_date'] = hospital_data['Admission_date'].map(lambda x:100*x.year + x.month)
hospital_data['Admission_date'].head()


# In[183]:


hospital_data['Discharge_date'] = pd.to_datetime(hospital_data['Discharge_date'], format='%Y-%m-%d')
hospital_data['Discharge_date'].dtype


# In[184]:


hospital_data['Discharge_date'] = hospital_data['Discharge_date'].map(lambda x:100*x.year + x.month)
hospital_data['Discharge_date'].head()


# In[185]:


diagnosis_data.isnull().sum()


# In[186]:


for col in diagnosis_data.columns.values:
    print(col, diagnosis_data[col].unique().shape)


# __Removing columns with only one unique value__

# In[187]:


diagnosis_data = diagnosis_data.drop(columns=['acetohexamide', 'metformin.rosiglitazone'])
diagnosis_data.shape


# In[188]:


diagnosis_data['num_medications'].unique()


# In[189]:


diagnosis_data['num_medications'].max()


# In[190]:


diagnosis_data['num_medications'] = pd.cut(diagnosis_data['num_medications'], bins=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])


# __Separating Test data__

# In[191]:


patient_test_data = patient_data[patient_data['istrain'] == 0]
patient_test_data.shape


# In[192]:


patient_data = patient_data[patient_data['istrain'] == 1]
patient_data.shape


# In[193]:


hospital_test_data = hospital_data[hospital_data['istrain'] == 0]
print(hospital_test_data.shape)
hospital_data = hospital_data[hospital_data['istrain'] == 1]
print(hospital_data.shape)
diagnosis_test_data = diagnosis_data[diagnosis_data['istrain'] == 0]
print(diagnosis_test_data.shape)
diagnosis_data = diagnosis_data[diagnosis_data['istrain'] == 1]
print(diagnosis_data.shape)


# In[194]:


test_data = patient_test_data.merge(hospital_test_data.merge(diagnosis_test_data, on='patientID'), on='patientID')
test_data.shape


# In[195]:


data = patient_data.merge(hospital_data.merge(diagnosis_data, on='patientID'), on='patientID')
data.shape


# __removing useless columns__

# In[196]:


data = data.drop(columns=['istrain'])
if 'istrain_x' in data:
    data = data.drop(columns=['istrain_x'])
if 'istrain_y' in data:
    data = data.drop(columns=['istrain_y'])
data.shape


# In[197]:


sns.countplot(data.loc[data['Target'] == 'Yes', 'admission_source_id'])


# In[198]:


sns.countplot(data.loc[data['Target'] == 'No', 'admission_source_id'])


# In[199]:


data.loc[(data['Target'] == 1) & (data['admission_source_id'] > 20)].shape


# __for values between 10 - 17 and values greater than 20 target is always 0__

# In[200]:


sns.countplot(data.loc[data['Target'] == 'Yes', 'admission_type_id'])


# In[201]:


sns.countplot(data.loc[data['Target'] == 'No', 'admission_type_id'])


# # Label Encoding

# In[202]:


from sklearn.preprocessing import LabelEncoder


# __No use for patientID and AdmissionID in the data, removing those__

# In[203]:


data = data.drop(columns=['patientID', 'AdmissionID'])
data.shape


# In[204]:


for col in data.columns.values:
    if data[col].dtype == 'object':
        print(col, data[col].unique().shape)


# __Ordinal columns from above columns:__
# race, gender, age, medical_specialty and all the meds from diagnosis table

# In[205]:


for col in data.columns.values:
    if data[col].dtype != 'object':
        print(col, data[col].unique().shape)


# In[206]:


le = LabelEncoder()
ordinal_cols = ['race', 'gender', 'age', 'Target', 'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide.metformin', 'glipizide.metformin', 'metformin.pioglitazone', 'change', 'diabetesMed']
for col in ordinal_cols:
    data[col] = le.fit_transform(data[col].astype(str))


# # Base Model

# In[207]:


X = data[data.columns.difference(['Target'])]
y = data['Target']


# In[208]:


X.head()


# In[209]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

rfc = RandomForestClassifier()
scores = cross_val_score(rfc, X, y, scoring='accuracy', cv=5)
print(scores)
scores.mean()


# In[210]:


y.value_counts()


# In[211]:


null_accuracy = 17786/(17786 + 6470)
null_accuracy


# In[212]:


def isGoodModel(scores):
    print(scores)
    print(scores.mean(), scores.mean() > null_accuracy)


# In[213]:


X.columns.values


# __Trying logistic regression__

# In[214]:


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
scores = cross_val_score(lr, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# __choosing logistic regression for base model__

# # Feature Engineering and Feature Selection

# In[215]:


for col in data.columns.values:
    print(col, 'Yes\n', data.loc[(data['Target'] == 1), col].value_counts())
    print(col, 'No\n', data.loc[(data['Target'] == 0), col].value_counts())


# In[216]:


for col in data.columns.values:
    print(col, data[col].unique().shape)


# __calculating chi2 value for features__

# In[217]:


from sklearn.feature_selection import chi2

X = data[data.columns.difference(['Target'])]
y = data['Target']
correlation_matrix = chi2(X, y)


# In[218]:


p_vals = correlation_matrix[1]
for i in range(len(X.columns.values)):
    print(X.columns.values[i], p_vals[i])


# In[219]:


p_vals = correlation_matrix[1]
unrelated_cols = []
for i in range(len(X.columns.values)):
    if p_vals[i] < 0.05:
        print(X.columns.values[i], p_vals[i])
    else:
        unrelated_cols.append(X.columns.values[i])


# In[220]:


X = data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']]
y = data['Target']

lr = LogisticRegression()
scores = cross_val_score(lr, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# __accuracy did not change__
# so removing the other columns from data.

# In[221]:


data = data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications', 'Target']]
data.shape


# In[222]:


data.head()


# # Model Selection

# In[223]:


from sklearn.svm import LinearSVC
lsvc = LinearSVC()
scores = cross_val_score(lsvc, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# In[225]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
scores = cross_val_score(knn, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# In[226]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=250)
scores = cross_val_score(abc, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# In[227]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(loss='deviance', n_estimators=200)
scores = cross_val_score(gbc, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# In[228]:


data['num_medications'] = le.fit_transform(data['num_medications'])
X = data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']]
y = data['Target']


# In[229]:


from xgboost import XGBClassifier
xgbc = XGBClassifier()
scores = cross_val_score(xgbc, X, y, scoring='accuracy', cv=5)
isGoodModel(scores)


# __AdaBoostClassifier and XGBoostClassifier__ by far has the best accuracy

# # Model Tuning

# In[230]:


from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [100, 150, 200, 250, 300]
}

grid_search = GridSearchCV(AdaBoostClassifier(), params, scoring='accuracy', cv=3)
grid_search.fit(X, y)
print(grid_search.best_score_, grid_search.best_params_)
rfc_cv_results = pd.DataFrame(grid_search.cv_results_)


# In[231]:


from sklearn.tree import DecisionTreeClassifier

params = {
    "base_estimator__criterion" : ["gini", "entropy"],
    "base_estimator__splitter" :   ["best", "random"],
}

dtc = DecisionTreeClassifier()
abc = AdaBoostClassifier(base_estimator = dtc)
grid_search = GridSearchCV(abc, param_grid=params, scoring = 'accuracy', cv=3)
grid_search.fit(X, y)
print(grid_search.best_score_, grid_search.best_params_)
abc_cv_results = pd.DataFrame(grid_search.cv_results_)


# In[232]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

abc = AdaBoostClassifier(n_estimators=250)

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
sss.get_n_splits(X, y)
scores = []; i=1; total_score = 0
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    abc.fit(X_train, y_train)
    y_pred = abc.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    scores.append(score)
    total_score += score
    print(i, 'iter complete')
    i += 1
total_score/5


# In[233]:


scores


# In[234]:


test_data.head()


# In[235]:


final_test = test_data[['age', 'change', 'diabetesMed', 'num_diagnoses', 'num_medications']]
final_test.head()


# In[236]:


data.head()


# In[238]:


for col in ['age', 'change', 'diabetesMed']:
    final_test[col] = le.fit_transform(final_test[col].astype(str))
final_test.head()


# In[240]:


test_data_predictions = abc.predict(final_test)


# In[254]:


final_output['patientID'] = test_data['patientID'].astype(str)
final_output['Target'] = test_data_predictions
final_output = final_output[['patientID', 'Target']]
final_output.head()


# In[257]:


file_name = 'E:\\workspace\\python\\scikitlearn\\onbaording2\\ML-onboarding\\Patient Readmission Assignment\\final_output.csv'
final_output.to_csv(file_name, sep='\t')


# In[ ]:




