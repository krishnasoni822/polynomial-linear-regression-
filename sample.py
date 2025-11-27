import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df1= pd.read_csv("titanic (4).csv")
df=pd.read_csv("sampledata.csv")
  

x= df[["Temperature (C)"]]
y= df[["Humidity"]]

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)

poly =PolynomialFeatures(degree=4)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.fit_transform(X_test)

lr=LinearRegression()
lr.fit(X_train_poly,Y_train)

y_pred=lr.predict(X_test_poly)
print(r2_score(Y_test,y_pred))
print(r2_score(Y_test,y_pred)*100)


