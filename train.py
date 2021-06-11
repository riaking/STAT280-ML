import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

MODEL = 'random-forest'

new_data = pd.DataFrame(pd.read_csv('Outputs/TaggedData_per5secs.csv'))

sc = StandardScaler()
ip = new_data['ip'].values
new_data['ip'] = sc.fit_transform(ip.reshape(-1, 1))
new_data.ip.describe()

x = new_data.drop(['click_time', 'attributed_time', 'is_attributed', 'seconds_5', 'hm_seconds_5', 'isFraud'], axis=1)
y = new_data['isFraud'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

rf = RandomForestClassifier(max_depth = 4)
rf.fit(x_train, y_train)

joblib.dump(rf, MODEL + '.joblib')
print("Done")
