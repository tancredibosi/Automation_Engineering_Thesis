import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


cell_df = pd.read_csv('cell_samples.csv')
cell_df.tail()
cell_df.shape
cell_df.size
cell_df.count()
cell_df['Class'].value_counts()


#Distribution of the classes
benign_df = cell_df[cell_df['Class']==2][0:200]
malignant_df = cell_df[cell_df['Class']==4][0:200]

axes = benign_df.plot(kind='scatter', x='Clump', y='UnifSize', color='blue', label='Benign')
malignant_df.plot(kind='scatter', x='Clump', y='UnifSize', color='red', label='Malignant', ax=axes)

plt.show(block=True)


#Identifying unwanted rows
cell_df.dtypes

cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors ='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes


#Remove unwanted rows
cell_df.columns

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
                      'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

# cell_df 100 rows and 11 columns
# picked 9 columns out of 11

# Independent variable
X = np.asarray(feature_df)

# dependent variable
y = np.asarray(cell_df['Class'])

y[0:5]


#Divide the data as Train/Test dataset

# cell_df(100 rows) --> Train(80 rows)/Test(20 rows)        based on number of rows
# Train (x, y)    x itself is a 2D array, y is 1D
#Test (x, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# 546 x 9
X_train.shape

# 546 x 1
y_train.shape

# 137 x 9
X_test.shape

# 137 x 1
y_test.shape


#Modeling (LogisticRegression with Scikit-learn)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

#Evaluation (results)

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))