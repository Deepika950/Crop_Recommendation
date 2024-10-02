import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#1step
data=pd.read_csv("Crop_recommendation.csv")
print(data.head())

#analysis of dataset
import matplotlib.pyplot as plt
vc=data['label'].value_counts()
plt.bar(vc.index,vc.values)
plt.show()

#for converting label datatype from object to numeric(number)datatype
from sklearn.preprocessing import LabelEncoder
#create a blank label encoder
le=LabelEncoder()
data['label']=le.fit_transform(data['label'])

print(data.info())
print(data.describe())
print(data.corr())
print(data.shape)


x=data['temperature']
y=data['label']
plt.scatter(x,y)
plt.show()


m=data['ph']
n=data['label']
plt.scatter(n,m)
plt.show()
plt.hist(x)
plt.show()


#2nd step(model training--1.splitting,2.loading,3.modeling)
a=data.iloc[:,0:7].values#use this alsoinstead of 7 [:,:-1]
print(a)
b=data.iloc[:,-1].values
print(b)

xtrain,xtest,ytrain,ytest=train_test_split(a,b,test_size=0.2)


#create a blank model
from sklearn.svm import SVC
model=SVC()
model.fit(xtrain,ytrain)
pred=model.predict(xtest)

from sklearn.metrics import accuracy_score,precision_score,recall_score

acc=accuracy_score(ytest,pred)
print(acc)
print(acc*100)

pre=precision_score(ytest,pred,average='micro')
print(pre)
print(pre*100)

rec=recall_score(ytest,pred,average='macro')
print(rec)
print(rec*100)

#for graphical representation
x1=["Accuracy","Precision","Recall"]
y1=[acc*100,pre*100,rec*100]
plt.bar(x1,y1)
plt.title("model performance metrics")
plt.show()


#testing for new input
xnew=[[60,55,44,23.00445915,82.3207629,7.840207144,263.9642476]]#for row and columns[[]]
output=model.predict(xnew)
print(output)