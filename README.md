# Digit-Recognizer
Our  goal is to correctly identify digits from a data set of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

Algorithms used are PCA and neural networks.


USE OF PCA:

There are many features in this data resulting in high dimensionality. PCA is used to compress the features into a small but 
informative set of features before using the data in a machine learning model. Data is normalized before PCA is applied. 
This is so the scale of the data does not throw of the PCA, and so the 0's are represented meaningfully. 
There is unequal variance in this data, and features with larger variance will influence the PCA more, creating bias. 
This is why the data is normalized.



