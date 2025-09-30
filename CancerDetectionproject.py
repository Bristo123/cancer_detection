#!/usr/bin/env python
# coding: utf-8

# In[259]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[221]:


cancer_data = pd.read_csv('The_Cancer_data_1500_V2.copy.csv')
cancer_data.head()


# In[222]:


cancer_dataset = cancer_data.copy(deep = True)
cancer_dataset.head(10)


# In[223]:


cancer_dataset.tail()


# In[224]:


cancer_dataset.info()


# In[225]:


cancer_dataset.shape


# In[226]:


cancer_dataset.describe()


# In[227]:


cancer_dataset.isna().sum()


# In[228]:


med = cancer_dataset['BMI'].median()
cancer_dataset['BMI'] = cancer_dataset['BMI'].fillna(med)
cancer_dataset.head()


# In[229]:


cancer_dataset.isna().sum()


# In[230]:


mod = cancer_dataset['PhysicalActivity'].mode()
cancer_dataset['PhysicalActivity'] = cancer_dataset['PhysicalActivity'].fillna(mod)
cancer_dataset.head(10)


# In[231]:


cancer_dataset.isna().sum()


# In[232]:


mean = cancer_dataset['AlcoholIntake'].mean()
cancer_dataset['AlcoholIntake'] = cancer_dataset['AlcoholIntake'].fillna(mean)
cancer_dataset.head()


# In[233]:


cancer_dataset.isna().sum()


# In[234]:


cancer_dataset.dropna
cancer_dataset.head()


# In[235]:


cancer_dataset.isna().sum()


# In[236]:


cancer_dataset.dropna(inplace = True)


# In[237]:


cancer_dataset.isna().sum()


# In[238]:


le = LabelEncoder()
x = ["Smoking","GeneticRisk","CancerHistory","Gender"]
for y in x:
    cancer_dataset[y] = le.fit_transform(cancer_dataset[y])

cancer_dataset.head()


# In[239]:


mapping = {"No Cancer":0, "Cancer":1}
cancer_dataset['Diagnosis'] = cancer_dataset['Diagnosis'].map(mapping)


# In[240]:


cancer_dataset.head()


# In[241]:


cancer_dataset['Diagnosis'].value_counts()


# In[242]:


x = cancer_dataset.drop(columns = 'Diagnosis', axis = True)
y = cancer_dataset['Diagnosis']


# In[243]:


print(x)


# In[244]:


print(y)


# In[245]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,stratify = y,random_state = 42)


# In[246]:


print(x.shape,x_train.shape,x_test.shape)


# In[247]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[248]:


pred = model.predict(x_test)


# In[249]:


confusion_matrix(y_test,pred)


# In[250]:


accuracy_score(y_test,pred)


# In[251]:


input_data = (62,0,35.479721,0,1,5.356890,2.414110,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0] == 0:
    print("The person does not have Cancer")
else:
    print("The Person have Cancer")


# In[252]:


model = KNeighborsClassifier(n_neighbors = 3)
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))


# In[253]:


input_data = (71,1,30.828784,0,2,9.361630,3.519683,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction[0] == 0:
    print("The Person does not have Cancer")
else:
    print("The Person has Cancer")


# In[254]:


model = DecisionTreeClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))


# In[255]:


input_data = (31,1,33.447125,0,2,1.668297,2.280636,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
if prediction[0]==0:
    print("The Person Does not have Cancer")
else:
    print("The Person has Cancer")


# In[257]:


model = RandomForestClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))


# In[258]:


input_data = (55,1,25.568216,0,1,7.795317,1.986138,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
if prediction[0]==0:
    print("The Person does not have Cancer")
else:
    print("The Person has Cancer")


# In[260]:


model = SVC(kernel = "linear")
model.fit(x_train,y_train)
pred = model.predict(x_test)
print(confusion_matrix(y_test,pred))
print(accuracy_score(y_test,pred))


# In[264]:


input_data = (67,0,23.663104,0,1,2.525860,2.414110,1)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = model.predict(input_data_reshaped)
if prediction[0]==0:
    print("The Person does not have Cancer")
else:
    print("The Person has Cancer")


# 
