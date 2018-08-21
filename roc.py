import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split,cross_val_predict
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve
from sklearn import metrics

clsr_names=["Nearest Neighbors",
            "Decision Tree", "Random Forest", "Logistic Regression"]

classifiers = [KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    LogisticRegression()]
#Read data in DataFrame
df01=pd.read_csv('adult.data', sep=',',header=0)
#Defined the column names
df01.columns= ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
df02=df01.dropna(axis=1, how='all')
df=df02.dropna(axis=0, how='any')
cols=df.dtypes
colnms=df.columns
#Check data types for all columns
print(cols)
print(colnms)
#Identify character varibles and process these variables by One Hot Encoding
i=0
cat_cols=[]
for eachcol in cols:
    if eachcol.name=="object":
        cat_cols.append(colnms[i])
    i+=1
print(cat_cols)
#Encoded all character variables
df1=pd.get_dummies(df,columns=cat_cols)
n=len(df1.index)
m=len(df1.columns)
print(df1.columns)
print("n-index= "+str(n))
print("m-columns= "+str(m))
#Select row(x) and colunms(y) from dataframe for modeling
x_all=df1.iloc[:,0:(m-2)]
y_all=df1.iloc[:,-1]
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, test_size=.5,random_state=0)
#Compute ROC curve and ROC area for Random Forest
model=classifiers[2]
model.fit(X_train, y_train)
predictions=model.predict_proba(X_test)
print("This is predictions-----")
print(predictions)
fpr1, tpr1, thresholds1 = metrics.roc_curve(y_test, predictions[:,1], pos_label=1)
auc1=metrics.auc(fpr1,tpr1)
print(auc1)
res1=pd.DataFrame({'FP':fpr1,'TP':tpr1,'Cut':thresholds1})
print(res1.head(10))
#Compute ROC curve and ROC area for Logistic Regression
model=classifiers[3]
model.fit(X_train, y_train)
predictions=model.predict_proba(X_test)
print("This is predictions-----")
print(predictions)
fpr2, tpr2, thresholds2 = metrics.roc_curve(y_test, predictions[:,1], pos_label=1)
auc2=metrics.auc(fpr2,tpr2)
print(auc2)
res2=pd.DataFrame({'FP':fpr2,'TP':tpr2,'Cut':thresholds2})
print(res2.head(10))
#Compute ROC curve and ROC area for Nearest Neighbors
model=classifiers[0]
model.fit(X_train, y_train)
predictions=model.predict_proba(X_test)
print("This is predictions-----")
print(predictions)
fpr3, tpr3, thresholds3 = metrics.roc_curve(y_test, predictions[:,1], pos_label=1)
auc3=metrics.auc(fpr3,tpr3)
print(auc3)
res3=pd.DataFrame({'FP':fpr3,'TP':tpr3,'Cut':thresholds3})
print(res3.head(10))
#Visualize ROC curves and AUC
plt.clf()
fig = plt.figure(figsize=(12, 6))
plt.plot(fpr1, tpr1,color='darkorange',label='Random Forest (area=%0.2f)'%auc1)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.plot(fpr2, tpr2,color='deeppink',label='Logistic Regression (area=%0.2f)'%auc2)
plt.plot(fpr3, tpr3,color='aqua',label='Nearest Neighbors (area=%0.2f)'%auc3)
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Performance Analysis - Receiver Operating Characteristic (ROC) Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.tight_layout()
fig.savefig('plot_roc.pdf',dpi=400)
