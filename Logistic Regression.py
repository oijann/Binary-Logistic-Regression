import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

Data = pd.read_csv ("diabetes2.csv")
Data.head().transpose()

Data.describe()

from sklearn.model_selection import train_test_split
X= Data.drop("Outcome", axis=1)
Y= Data[["Outcome"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state=7)

model = LogisticRegression()
model.fit(X_train, Y_train)
Y_predict= model.predict(X_test)
model_score= model.score(X_test, Y_test)

print(model_score)
print(metrics.confusion_matrix(Y_test, Y_predict))

Y_predict=model.predict([[1,84,65,30,0,25.6,0.352,30]])
print(Y_predict)