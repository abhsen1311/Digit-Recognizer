import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve,roc_auc_score,auc

dataset=pd.read_csv("C:/Users/Abhiijit/Desktop/creditcard.csv")

dataset.head()
#print first 5 lines
dataset.head(5)
#print data shape
print(dataset.shape)

dataset.head()
dataset.isna()

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
dataset['normalized amount']=sc.fit_transform(dataset['Amount'].values.reshape(-1,1))
dataset=dataset.drop(['Amount'],axis=1)

dataset.head()
 