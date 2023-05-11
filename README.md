# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset.
2.Check for null and duplicate values.
3.Assign x and y values.
4.Split data into train and test data.
5.Import logistic regression and fit the training data.
6.Predict y value.
7.Calculate accuracy and confusion matrix.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DHANUSH.G.R.
RegisterNumber:212221040038  
*/
```
import pandas as pd

data=pd.read_csv('/content/Placement_Data.csv')

print("Placement data:")

data.head()


data1=data.copy()

data1=data1.drop(["sl_no","salary"],axis=1)#removes the specified row or coloumn

print("Salary data:")

data1.head()

print("Checking the null() function:")

data1.isnull().sum()

print ("Data Duplicate:")

data1.duplicated().sum()

print("Print data:")

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

data1["gender"]=le.fit_transform(data1["gender"])

data1["ssc_b"]=le.fit_transform(data1["ssc_b"])

data1["hsc_b"]=le.fit_transform(data1["hsc_b"])

data1["hsc_s"]=le.fit_transform(data1["hsc_s"])

data1["degree_t"]=le.fit_transform(data1["degree_t"])

data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])

data1["status"]=le.fit_transform(data1["status"])

data1


print("Data-status value of x:")

x=data1.iloc[:,:-1]

x


print("Data-status value of y:")

y=data1["status"]

y


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


print ("y_prediction array:")

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver="liblinear")#a library for large

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

y_pred


from sklearn.metrics import accuracy_score 

accuracy=accuracy_score(y_test,y_pred) 

print("Accuracy value:")

accuracy


from sklearn.metrics import confusion_matrix 

confusion=(y_test,y_pred) 

print("Confusion array:")

confusion


from sklearn.metrics import classification_report 

classification_report1=classification_report(y_test,y_pred) 

print("Classification report:")

print(classification_report1)



print("Prediction of LR:")

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/32d0da15-2334-4be9-b1e1-632c4bdbb3e8)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/e0d0476e-3f2f-4bd1-8e32-64c7284a56e6)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/d209608d-4fd9-42b6-9ebd-803effa4e656)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/86099f5a-475b-4334-bccf-f02618640632)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/446d036c-3e7e-45cf-85b4-88c7b838b509)


![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/68a4e8ee-2ca6-4eed-88ef-6a32607f2185)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/466aec79-dad2-422d-9e66-9857cc69599e)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/c68ea5df-6267-4d38-9ee5-a8b68ccadfbf)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/093ae3fc-a3b6-4599-895d-484ee9c1047c)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/566f6774-7b4e-4797-81d2-8f92d6fb901c)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/7f01d323-91f5-4271-aadc-b31520b31534)

![image](https://github.com/Dhanush12022004/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128135558/50e8ede8-d5d0-4033-9116-d53218d05e8c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
