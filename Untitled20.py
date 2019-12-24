#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[130]:


df= pd.read_csv(r"C:\\Users\\Prabhat Singh\desktop\dataset\heart.csv")


# In[131]:


# first five rows
df.head()


# The above data contain 
# Data contains
# 
# 1)age - age in years
# 
# 2)sex - (1 = male; 0 = female)
# 
# 3)cp - chest pain type
# 
# 4)trestbps - resting blood pressure (in mm Hg on    admission to the hospital)
# 
# 5)chol - serum cholestoral in mg/dl
# 
# 6)fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 
# 7)restecg - resting electrocardiographic results
# 
# 8)thalach - maximum heart rate achieved
# 
# 9)exang - exercise induced angina (1 = yes; 0 = no)
# 
# 10)oldpeak - ST depression induced by exercise relative to rest
# 
# 11)slope - the slope of the peak exercise ST segment
# 
# 12)ca - number of major vessels (0-3) colored by flourosopy
# 
# 13)thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
# 
# 14)target - have disease or not (1=yes, 0=no)
# 

# In[132]:


df.target.value_counts()


# In[133]:


sns.countplot(x="target",data=df,palette="bwr")
plt.show()


# In[134]:


countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Have no Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))


# In[135]:


sns.countplot(data=df,x="sex",palette="mako_r")
plt.xlabel("sex (female=0 ,male=1)")
plt.show()


# In[136]:


countFemale = len(df[df.sex==0])
countMale = len(df[df.sex==1])
print("numer of male:{:.2f}%".format((countFemale / (len(df.sex))*100)))
print("numer of male:{:.2f}%".format((countMale/ (len(df.sex))*100)))


# In[137]:


df.groupby("target").mean()


# In[138]:


pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[139]:


pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(10,8))
plt.title("heart diesease by sex")
plt.xlabel("sex")
plt.ylabel("frequency")
plt.legend(["have disease", "no disease" ])
plt.xticks(rotation=0)
plt.show()


# In[140]:


plt.scatter(x=df.age[df.target==1],y=df.thalach[(df.target==1)],c="red")
plt.scatter(x=df.age[df.target==0],y=df.thalach[(df.target==0)])
plt.xlabel("age")
plt.ylabel("maximum heart rate")
plt.legend(["disease", "no disease"])
plt.show()
            


# In[141]:


pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency by Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()


# In[142]:


pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6))
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()


# Creating Dummy Variables
# 
# Since 'cp', 'thal' and 'slope' are categorical variables we'll turn them into dummy variables.
# 

# In[143]:


a = pd.get_dummies(df["cp"],prefix="cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")


# In[144]:


frames=[df,a,b,c]
df=pd.concat(frames,axis=1)


# In[145]:


df.head()


# In[146]:


df=df.drop(columns=["cp","thal","slope"],axis=1)


# In[147]:


df.head()


# ### logistic regression
# 

# In[148]:


y=df.target.values
x_data=df.drop(["target"],axis=1)


# In[149]:


x


# ### normalize data
# 
# ![image.png](attachment:image.png)

# In[150]:


x=(x_data-np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values


# In[151]:


x


# In[152]:


x_train, x_test,y_train, y_test =train_test_split(x,y,test_size=20,random_state=0)


# In[153]:


#transpose matrices
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T


# In[155]:


lr=LogisticRegression()
lr.fit(x_train.T,y_train.T)


# In[156]:


accuracy= lr.score(x_test.T,y_test.T)*100


# In[159]:


print("Accuracy for logistic regression is {:.2f}%".format(accuracy) )


# ### K-nearest neighbour(KNN)

# In[174]:


# for best k value
np.sqrt(len(y_test))


# In[175]:


from sklearn.neighbors import KNeighborsClassifier


# In[224]:


knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train.T,y_train.T)


# In[225]:


prediction=knn.predict(x_test.T)


# In[226]:


print ("Accuracy for knn is {:.2f}%".format(knn.score(x_test.T,y_test.T)*100))


# ### Support Vector Machine (SVM) Algorithm

# In[227]:



from sklearn.svm import SVC


# In[241]:


svm = SVC(random_state=1,gamma='auto')
svm.fit(x_train.T,y_train.T)
acc=svm.score(x_test.T,y_test.T)*100
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))




# ### Naive Bayes Algorithm

# In[242]:


from sklearn.naive_bayes import GaussianNB


# In[246]:


nb=GaussianNB()
nb.fit(x_train.T, y_train.T)
acc=nb.score(x_test.T,y_test.T)*100
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))


# ### Decision Tree Algorithm

# In[247]:


from sklearn.tree import DecisionTreeClassifier


# In[248]:


dtc = DecisionTreeClassifier()


# In[249]:


dtc.fit(x_train.T, y_train.T)


# In[251]:


acc = dtc.score(x_test.T, y_test.T)*100


# In[252]:


print("Decision Tree Test Accuracy {:.2f}%".format(acc))


# ### Random Forest Classification

# In[253]:


from sklearn.ensemble import RandomForestClassifier


# In[270]:


rf = RandomForestClassifier(n_estimators = 2000, random_state = 1)


# In[271]:


rf.fit(x_train.T, y_train.T)


# In[272]:


acc = rf.score(x_test.T,y_test.T)*100


# In[273]:


print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))


# #### Confusion Matrix

# In[278]:


# Predicted values
y_head_lr = lr.predict(x_test.T)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn3.fit(x_train.T, y_train.T)
y_head_knn = knn3.predict(x_test.T)
y_head_svm = svm.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)
y_head_rf = rf.predict(x_test.T)


# In[279]:


from sklearn.metrics import confusion_matrix


# In[280]:


cm_lr = confusion_matrix(y_test,y_head_lr)
cm_knn = confusion_matrix(y_test,y_head_knn)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)


# In[283]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})


plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()


# In[ ]:




