import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# including data type in the file
weather = pd.read_csv('weatherAUS.csv')

# including mean to replace NAN file
weather['MinTemp'].fillna(weather['MinTemp'].mean(),inplace=True)
weather['MaxTemp'].fillna(weather['MaxTemp'].mean(),inplace=True)
weather['Rainfall'].fillna(weather['Rainfall'].mean(),inplace=True)
weather['Sunshine'].fillna(weather['Sunshine'].mean(),inplace=True)
weather['WindGustDir'].fillna('W',inplace=True)
weather['WindGustSpeed'].fillna(weather['WindGustSpeed'].mean(),inplace=True)
weather['WindDir9am'].fillna('SE',inplace=True)
weather['WindDir3pm'].fillna('SE',inplace=True)
weather['Humidity9am'].fillna(weather['Humidity9am'].mean(),inplace=True)
weather['Humidity3pm'].fillna(weather['Humidity3pm'].mean(),inplace=True)
weather['Pressure9am'].fillna(weather['Pressure9am'].mean(),inplace=True)
weather['Pressure3pm'].fillna(weather['Pressure3pm'].mean(),inplace=True)
weather['WindSpeed9am'].fillna(weather['WindSpeed9am'].mean(),inplace=True)
weather['WindSpeed3pm'].fillna(weather['WindSpeed3pm'].mean(),inplace=True)
weather['Cloud9am'].fillna(weather['Cloud9am'].mean(),inplace=True)
weather['Cloud3pm'].fillna(weather['Cloud3pm'].mean(),inplace=True)
weather['Temp9am'].fillna(weather['Temp9am'].mean(),inplace=True)
weather['Temp3pm'].fillna(weather['Temp3pm'].mean(),inplace=True)
weather.dropna(inplace=True)

# deleting the unwanted parameter
weather.drop(['Date', 'Location','WindDir3pm','WindDir9am','WindGustDir'],axis=1,inplace=True)


# changing string to number type
label_encoder = preprocessing.LabelEncoder()
weather['RainToday']=label_encoder.fit_transform(weather['RainToday'])
weather['RainTomorrow']=label_encoder.fit_transform(weather['RainTomorrow'])

# Using logistic regression to train the model
X=weather.drop(['RainTomorrow','RainToday'],axis=1)
y=weather['RainToday']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=101)
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train,y_train)

pickle.dump(lr, open('model.pkl','wb'))

# predicting the value
'''predict = lr.predict([[8.7	,21.0,	0.0 ,	2.4	,7.61	,24.0	,9.0,	9.0	,75.0,40.00,	1021.0,1017.6,	5.0,	2.0, 	13.8,	20.60,0	]])
print(predict)'''
