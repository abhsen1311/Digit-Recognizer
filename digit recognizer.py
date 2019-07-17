import numpy as np
import pandas as pd

np.random.seed(1)

train=pd.read_csv("C:/Users/Abhiijit/Desktop/train.csv")
test=pd.read_csv("C:/Users/Abhiijit/Desktop/test.csv")

print(train.shape)
print(test.shape)

train.head()

#This simple histogram shows the count of digits in the training data 
#for each number. This graphic is used to visualize if there is an 
#unequal sample size among the digits. The sample size for each digit 
#appear to be comparable. There is no issue of unequal sampling.

import matplotlib.pyplot as plt

plt.hist(train['label'])

plt.title("frequency histogram of numbers in training data")

plt.xlabel("number value")
plt.ylabel("frequency")     


import math

#plot the first 25 digits in the training set

f,ax=plt.subplots(5,5)

#plot some 4s as an example

for i in range(1,26):
    # create a 1024x1024x3 array of 8 bit unsigned integers
    data=train.iloc[i,1:785].values
    nrows,ncols=28,28
    grid=data.reshape((nrows,ncols))
    n=math.ceil(i/5)-1
    m=[0,1,2,3,4]*5
    ax[m[i-1],n].imshow(grid)
    
#There are many features in this data resulting in high dimensionality
#. PCA is used to compress the features into a small but 
#informative set of features before using the data in a machine learn
#ing model. Data is normalized before PCA is applied. This is so the 
#scale of the data does not throw of the PCA, and so the 0's are 
#represented meaningfully. There is unequal variance in this data, 
#and features with larger variance will influence the PCA more, 
#creating bias. This is why the data is normalized.    
    
    
#normalize the data

label_train=train['label']
train=train.drop('label',axis=1)

#normalize the data

train=train/255
test=test/255

train['label']=label_train

from sklearn import decomposition
from sklearn import datasets
import matplotlib.pyplot as plt


#PCA decomposition

pca=decomposition.PCA(n_components=200) #find first 200 PCs

pca.fit(train.drop('label',axis=1))

plt.plot(pca.explained_variance_ratio_)

plt.ylabel("% of variance explained")

#PCA reaches asymptote at 50 i.e= the optimal number of PCs to use

pca=decomposition.PCA(n_components=50)

pca.fit(train.drop('label',axis=1))

PCtrain=pd.DataFrame(pca.transform(train.drop('label',axis=1)))

PCtrain['label']=train['label']


#decompose test data

PCtest=decomposition.PCA(pca.transform(test))

from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

x=PCtrain[0]
y=PCtrain[1]
z=PCtrain[2]


colors=[int(i%9) for i in PCtrain['label']]
ax.scatter(x,y,z,c=colors,marker='o',label=colors)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()


from sklearn.neural_network import MLPClassifier
y=PCtrain['label'][0:20000]
X=PCtrain.drop('label',axis=1)[0:20000]

clf=MLPClassifier(solver='lbfgs',alpha=1e-5,random_state=1,hidden_layer_sizes=(3500,))

clf.fit(X,y)

from sklearn import metrics

predicted=clf.predict(PCtrain.drop('label',axis=1)[20001:42000])

expected=PCtrain['label'][20001:42000]

print("classification report for classifier %s:\n%s\n"%(clf,metrics.classification_report(expected,predicted)))

print("confusion matrix:\n%s"%metrics.confusion_matrix(expected,predicted))

output=pd.DataFrame(clf.predict(PCtest),columns=['Label'])

output.reset_index(inplace=True)
output.rename(columns={'index':'IMageId'},inplace=True)
output['ImageId']=output['ImageId']+1
output.to_csv("C:/Users/Abhiijit/Desktop/output.csv",index=False)

    
