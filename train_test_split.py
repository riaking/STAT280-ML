import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler

new_data = pd.read_csv('Outputs/june7_TaggedData_per5secs.csv')
#new_data = pd.read_csv('Data/train_sample.csv')
#new_data = new_data.sample(20000)

new_data = new_data.drop(['Unnamed: 0', 'click_time', 'attributed_time', 'is_attributed', 'hm_seconds_5', 'day'], axis=1)

[train, test] = train_test_split(new_data, test_size = 0.3, random_state = 123)
print("Split done")

#sc = StandardScaler()
#ip = new_data['ip'].values
#new_data['ip'] = sc.fit_transform(ip.reshape(-1, 1))
#new_data.ip.describe()

feature_cols = ['hour', 'minute', 'second',
                'ip', 'app', 'device',
                'os', 'channel']
x_train = train[feature_cols]
y_train = train['isFraud']

rus = RandomUnderSampler(random_state=123)
x_train, y_train = rus.fit_resample(x_train, y_train)
print("Resampling done")

cat_features = ['ip', 'app', 'device', 'os', 'channel']
count_enc = ce.CountEncoder(cols=cat_features)
count_enc.fit(x_train[cat_features])
x_train = x_train.join(count_enc.transform(x_train[cat_features]).add_suffix("_count"))

count_enc.fit(test[cat_features])
test = test.join(count_enc.transform(test[cat_features]).add_suffix("_count"))

feature_cols = ['hour', 'minute', 'second',
                'ip_count', 'app_count', 'device_count',
                'os_count', 'channel_count']
x_train = x_train[feature_cols]

test_cols = ['hour', 'minute', 'second',
                'ip_count', 'app_count', 'device_count',
                'os_count', 'channel_count', 'isFraud']
test = test[test_cols]
print("Preprocessing done")

x_train.to_csv('Data/x_train.csv')
y_train.to_csv('Data/y_train.csv')
test.to_csv('Data/test_.csv')
print("Generating csv files...")
print("Done")
