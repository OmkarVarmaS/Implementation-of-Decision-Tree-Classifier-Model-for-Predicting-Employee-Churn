# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Omkar Varma S
RegisterNumber:  212224240108
```

```

import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
print("data.info():")
data.info()
print("isnull() and sum():")
data.isnull().sum()
print("data value counts():")
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

```

## Output:

<img width="1569" height="255" alt="image" src="https://github.com/user-attachments/assets/55b344ca-33f1-4331-8cce-021c9ce7e466" />

<img width="609" height="370" alt="image" src="https://github.com/user-attachments/assets/8c2a338a-ac6a-41b9-958a-91fe0ea78f83" />

<img width="1516" height="472" alt="image" src="https://github.com/user-attachments/assets/b66c72b5-0ef0-4307-ac8f-34ca844cc95a" />

<img width="1554" height="214" alt="image" src="https://github.com/user-attachments/assets/35ff09dc-49c8-473e-ba75-8d557b1df2ec" />

<img width="1712" height="251" alt="image" src="https://github.com/user-attachments/assets/96132374-e203-4243-b36d-c3391459f135" />

<img width="1768" height="252" alt="image" src="https://github.com/user-attachments/assets/9953f264-0dfe-4e80-8e82-420f05410c05" />

<img width="1744" height="260" alt="image" src="https://github.com/user-attachments/assets/624182a5-e44d-4c6d-a219-93e807af7f0f" />

<img width="1764" height="79" alt="image" src="https://github.com/user-attachments/assets/33bc843e-2540-42d5-8a78-47096e5a7772" />

<img width="1662" height="125" alt="image" src="https://github.com/user-attachments/assets/f6f78909-5746-4e85-b29c-abe9946bdb17" />

<img width="914" height="669" alt="image" src="https://github.com/user-attachments/assets/cbf094b4-e662-44b8-a569-cb3f2de4b62b" />






## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
