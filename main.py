#%%

#96.5 Accuracy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
labelEncoder_Y = LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state= 0)
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy',random_state= 0)
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import os
from joblib import dump
df = pd.read_csv('./data.csv')
#print(df.shape)  # Checking the data row and columns

#print(df.isna().sum()) #Checking of all the empty columns
df = df.dropna(axis=1) #droping the last column

#print(df['diagnosis'].value_counts()) # checking the type
sns.countplot(df['diagnosis']) #making a visualization of the data
df.iloc[:,1] = labelEncoder_Y.fit_transform(df.iloc[:,1].values)  # 1 = M, 0 = B

sns.pairplot(df.iloc[:,1:5], hue='diagnosis') 
df.iloc[:1,1:12].corr() #seeing co-relation 
plt.figure(figsize=(10,10))
# os.cls()
print('\033c')
sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%') #visualising the co-relation

# X->independent Y- Dependent
X  = df.iloc[:,2:31].values
Y = df.iloc[:,1].values
# 75%-TRAINING || 25%-TESTING 
X_train,X_test,Y_train,Y_test = train_test_split(X, Y,test_size = 0.2,random_state = 0) # 75%-TRAINING || 25%-TESTING 

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# print(X_train)

#TRAINING PART VALIDATION
def models(X_train,Y_train):
    '''Function for training models '''
    # LOGISTIC REGRESSION
    log.fit(X_train,Y_train)
    #DECISION TREE CLASSIFIER
    tree.fit(X_train,Y_train)
    #RANDOM FOREST CLASSIFIER
    forest.fit(X_train,Y_train)

    print('[0] Logistic Regression Training Accuracy:',log.score(X_train,Y_train)*100)
    print('[1] Decision Tree Classifier Training Accuracy:',tree.score(X_train,Y_train)*100)
    print('[2] Random Forest Classifier Training Accuracy:',forest.score(X_train,Y_train)*100)
    return log,tree,forest

model = models(X_train,Y_train)

#TESTING PART 
'''1-st METHOD
 for i in range(len(model)):
    print('MODEL',i)
    cm = confusion_matrix(Y_test,model[i].predict(X_test))
    print(cm)
    TP = cm[0][0] #true_POSITIVE 
    TN = cm[1][1] #true_POSITIVE 
    FP = cm[1][0] #False_POSITIVE 
    FN = cm[0][1] #False_POSITIVE 
    print("Testing_Accuracy = ",(TP+TN)/(TP + TN + FN + FP))
    print()'''
for i in range(len(model)):
    print('MODEL',i)
    print(classification_report(Y_test,model[i].predict(X_test)))
    print(accuracy_score(Y_test,model[i].predict(X_test))*100)
#IN TRAINING DECISION WINS(100%) AND IN TESTING RANDOM FOREST(96.5%) 


pred = model[2].predict(X_test)
print(pred)
print()
print(Y_test)

dump(model[2],"breast_cancer_detection.joblib")
#%%
