# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import pickle

df = pd.read_csv('insurance.csv')


X=df.drop(columns=["charges"])

#Converting words to integer values
def convert_to_int1(word):
    word_dict = {'female':1,"male":0}
    return word_dict[word]

X['sex'] = X['sex'].apply(lambda x : convert_to_int1(x))

#Converting words to integer values
def convert_to_int2(word):
    word_dict1 = {'yes':1,"no":0}
    return word_dict1[word]

X['smoker'] = X['smoker'].apply(lambda y : convert_to_int2(y))

#Converting words to integer values
def convert_to_int3(word):
    word_dict2 = {'northeast':1,'northwest':2,'southeast':3,'southwest':4}
    return word_dict2[word]

X['region'] = X['region'].apply(lambda z : convert_to_int3(z))


Y=df["charges"]   

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X,Y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0,0,0,0,0,0]]))