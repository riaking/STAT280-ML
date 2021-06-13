import re
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
#import category_encoders as ce
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix, roc_auc_score, precision_score, recall_score

#MODEL = 'XGBClassifier'
#MODEL = 'MLPClassifier'
MODEL = 'random-forest'

model = load(MODEL + '/' + MODEL + '.joblib')

test = pd.DataFrame(pd.read_csv('Data/test_.csv'))
test = test.drop(['Unnamed: 0'], axis=1)
print(test.head())

## MODEL EVALUATION

feature_cols = ['hour', 'minute', 'second',
                'ip_count', 'app_count', 'device_count',
                'os_count', 'channel_count']

predictions = model.predict(test[feature_cols].fillna(0))
print(predictions)
print(any(predictions))
#predictions = [round(value) for value in predictions]

print("accuracy: ", accuracy_score(test['isFraud'], predictions) * 100)
print("auc: ", roc_auc_score(test['isFraud'], predictions) * 100)
print("precision_score: ", precision_score(test['isFraud'], predictions) * 100)
print("recall_score: ", recall_score(test['isFraud'], predictions) * 100)

# Output classification report to csv
cfReport = classification_report(test['isFraud'], predictions, output_dict=True)
cfDF = pd.DataFrame(cfReport).transpose()
cfDF.to_csv(MODEL + '/classification_report.csv')
#
## Output confusion matrix
#plot_confusion_matrix(model, test[feature_cols], test['isFraud'])
#plt.savefig(MODEL + '/confusionMatrix_test.png')
#plt.clf()

print("Done")
