import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#MODEL = 'XGBClassifier'
MODEL = 'MLPClassifier'
#MODEL = 'random-forest'

x_train = pd.DataFrame(pd.read_csv('Data/x_train.csv'))
y_train = pd.DataFrame(pd.read_csv('Data/y_train.csv'))

y_train = y_train.drop(['Unnamed: 0'], axis=1)
x_train = x_train.drop(['Unnamed: 0'], axis=1)

#rf = RandomForestClassifier(max_depth = 5)
#rf.fit(x_train, y_train)
#joblib.dump(rf, MODEL + '/' + MODEL + '.joblib')

#xgb = XGBClassifier(max_depth=10, subsample=0.9, tree_method='hist', max_bin = 300)
#xgb.fit(x_train, y_train)
#joblib.dump(xgb, MODEL + '/' + MODEL + '.joblib')

nn = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=280)
nn.fit(x_train, y_train)
joblib.dump(nn, MODEL + '/' + MODEL + '.joblib')

print("Training Done")
