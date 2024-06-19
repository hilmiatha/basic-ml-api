from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle as pkl

#load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train the model
model = RandomForestClassifier(n_estimators=10, random_state=42)

model.fit(X_train, y_train)

#predict the model
y_pred = model.predict(X_test)

#calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

#save the model
with open('model_basic_randomforest.pkl', 'wb') as model_basic_randomforest_pkl:
    pkl.dump(model, model_basic_randomforest_pkl)
    
    
print(X)