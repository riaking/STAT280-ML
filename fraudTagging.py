
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_FOLDER = 'Outputs'

# Read Training Set
train = pd.read_csv('Data/train.csv',parse_dates=['click_time'])

train['day'] = train['click_time'].dt.day.astype('uint8')
train['hour'] = train['click_time'].dt.hour.astype('uint8')
train['minute'] = train['click_time'].dt.minute.astype('uint8')
train['second'] = train['click_time'].dt.second.astype('uint8')
print("transform done")

train_2days = train[train['day'].isin([7])]
train_2days['minute'] = train_2days['minute'].apply(lambda x: '{0:0>2}'.format(x))

### per minute, using click-squared

#train_2days['hour_minute'] = train_2days['hour'].astype(str) + train_2days['minute'].astype(str)
#train_2days['hour_minute'] = train_2days['hour_minute'].astype('int32')
#print("transform done")

#conversion_minute = train_2days.groupby('hour_minute').aggregate(
#                        {
#                            'is_attributed':sum,
#                            'click_time':"count"
#                        }
#                    )
#
#conversion_minute['conversion_rate'] = conversion_minute['is_attributed']/conversion_minute['click_time']
#conversion_minute.reset_index(level=0, inplace=True)
#
#conversion_minute.plot(x ='hour_minute', y='conversion_rate', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/conversionRate_perMinute.png')
#plt.clf()
#
#df = conversion_minute
#df.loc[(df['is_attributed'] != 0), 'threshold'] = df['click_time']/df['conversion_rate']
#
#df.plot(x ='hour_minute', y='threshold', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/threshold_perMinute_clickSquared.png')
#plt.clf()
#
#Q1 = df.threshold.quantile(0.25)
#Q3 = df.threshold.quantile(0.75)
#IQR = Q3 - Q1
#threshold = Q3 + 1.5 * IQR
#print("Clicks squared per downloads threshold: ", threshold)
#
#df.loc[(df['is_attributed'] != 0) & (df['threshold'] > threshold), 'isFraud'] = 1
#df.loc[(df['isFraud'].isnull()), 'isFraud'] = 0
#
#new_data = pd.merge(train_2days, df[['hour_minute', 'isFraud']], on='hour_minute', how='left')
#event_rate = new_data.isFraud.sum()/len(new_data)
#print("Event rate per minute using click squared threshold: ", event_rate)
#new_data.to_csv('TaggedData_perMinute.csv')

### per 5 seconds, click-squared

#ranges = [-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, np.inf]  # np.inf for infinity
#labels = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
#train_2days['seconds_5'] = pd.cut(train_2days['second'],
#                                  bins=ranges,
#                                  labels=labels)
#train_2days['hm_seconds_5'] = train_2days['hour'].astype(str) + train_2days['minute'].astype(str) + train_2days['seconds_5'].astype(str)
#train_2days['hm_seconds_5'] = train_2days['hm_seconds_5'].astype('int32')
#
#conversion_seconds_5 = train_2days.groupby('hm_seconds_5').aggregate(
#                        {
#                            'is_attributed':sum,
#                            'click_time':"count"
#                        }
#                    )
#
#conversion_seconds_5['conversion_rate'] = conversion_seconds_5['is_attributed']/conversion_seconds_5['click_time']
#conversion_seconds_5.reset_index(level=0, inplace=True)
#
#conversion_seconds_5.plot(x ='hm_seconds_5', y='conversion_rate', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/june7_conversionRate_per5secs.png')
#plt.clf()
#
#df = conversion_seconds_5
#df.loc[(df['is_attributed'] != 0), 'threshold'] = df['click_time']/df['is_attributed']
#
#df.plot(x ='hm_seconds_5', y='threshold', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/june7_threshold_per5secs_clickSquared.png')
#plt.clf()
#
#co = conversion_seconds_5[conversion_seconds_5['conversion_rate']==0]
#Q1 = co.click_time.quantile(0.25)
#Q3 = co.click_time.quantile(0.75)
#IQR = Q3 - Q1
#click_outlier = Q3 + 1.5 * IQR
#print("Number of clicks threshold: ", click_outlier)
#
#Q1 = df.threshold.quantile(0.25)
#Q3 = df.threshold.quantile(0.75)
#IQR = Q3 - Q1
#threshold = Q3 + 1.5 * IQR
#print("Clicks squared per downloads threshold: ", threshold)
#
#df.loc[(df['is_attributed'] == 0) & (df['click_time'] > click_outlier), 'isFraud'] = 1
#df.loc[(df['is_attributed'] != 0) & (df['threshold'] > threshold), 'isFraud'] = 1
#df.loc[(df['isFraud'].isnull()), 'isFraud'] = 0
#
#new_data = pd.merge(train_2days, df[['hm_seconds_5', 'isFraud']], on='hm_seconds_5', how='left')
#event_rate = new_data.isFraud.sum()/len(new_data)
#print("Event rate per 5 seconds using click squared threshold: ", event_rate)
#new_data.to_csv(OUTPUT_FOLDER + '/june7_TaggedData_per5secs.csv')

### per 10 seconds, using click-squared

#ranges = [-1, 9, 19, 29, 39, 49, np.inf]  # np.inf for infinity
#labels = ['01', '02', '03', '04', '05', '06']
#train_2days['seconds_10'] = pd.cut(train_2days['second'],
#                                  bins=ranges,
#                                  labels=labels)
#train_2days['hm_seconds_10'] = train_2days['hour'].astype(str) + train_2days['minute'].astype(str) + train_2days['seconds_10'].astype(str)
#train_2days['hm_seconds_10'] = train_2days['hm_seconds_10'].astype('int32')
#print("transform done")
#
#conversion_seconds_10 = train_2days.groupby('hm_seconds_10').aggregate(
#                        {
#                            'is_attributed':sum,
#                            'click_time':"count"
#                        }
#                    )
#
#conversion_seconds_10['conversion_rate'] = conversion_seconds_10['is_attributed']/conversion_seconds_10['click_time']
#conversion_seconds_10.reset_index(level=0, inplace=True)
#
#conversion_seconds_10.plot(x ='hm_seconds_10', y='conversion_rate', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/conversionRate_per10secs.png')
#plt.clf()
#
#df = conversion_seconds_10
#df.loc[(df['is_attributed'] != 0), 'threshold'] = df['click_time']/df['is_attributed']
#
#df.plot(x ='hm_seconds_10', y='threshold', kind = 'line')
#plt.savefig(OUTPUT_FOLDER + '/threshold_per10secs_clickSquared.png')
#plt.clf()
#
#co = conversion_seconds_10[conversion_seconds_10['conversion_rate'] == 0]
#Q1 = co.click_time.quantile(0.25)
#Q3 = co.click_time.quantile(0.75)
#IQR = Q3 - Q1
#click_outlier = Q3 + 1.5 * IQR
#print("Number of clicks threshold: ", click_outlier)
#
#Q1 = df.threshold.quantile(0.25)
#Q3 = df.threshold.quantile(0.75)
#IQR = Q3 - Q1
#threshold = Q3 + 1.5 * IQR
#print("Clicks squared per downloads threshold: ", threshold)
#
#df.loc[(df['is_attributed'] == 0) & (df['click_time'] > click_outlier), 'isFraud'] = 1
#df.loc[(df['is_attributed'] != 0) & (df['threshold'] > threshold), 'isFraud'] = 1
#df.loc[(df['isFraud'].isnull()), 'isFraud'] = 0
#
#new_data = pd.merge(train_2days, df[['hm_seconds_10', 'isFraud']], on='hm_seconds_10', how='left')
#event_rate = new_data.isFraud.sum()/len(new_data)
#print("Event rate per 10 seconds using click squared threshold: ", event_rate)
#new_data.to_csv(OUTPUT_FOLDER + '/TaggedData_per10secs.csv')

### per 15 seconds, click-squared

ranges = [-1, 14, 29, 44, np.inf]  # np.inf for infinity
labels = ['01', '02', '03', '04']
train_2days['seconds_15'] = pd.cut(train_2days['second'],
                                  bins=ranges,
                                  labels=labels)
train_2days['hm_seconds_15'] = train_2days['hour'].astype(str) + train_2days['minute'].astype(str) + train_2days['seconds_15'].astype(str)
train_2days['hm_seconds_15'] = train_2days['hm_seconds_15'].astype('int32')

conversion_seconds_15 = train_2days.groupby('hm_seconds_15').aggregate(
                        {
                            'is_attributed':sum,
                            'click_time':"count"
                        }
                    )

conversion_seconds_15['conversion_rate'] = conversion_seconds_15['is_attributed']/conversion_seconds_15['click_time']
conversion_seconds_15.reset_index(level=0, inplace=True)

conversion_seconds_15.plot(x ='hm_seconds_15', y='conversion_rate', kind = 'line')
plt.savefig(OUTPUT_FOLDER + '/conversionRate_per15secs.png')
plt.clf()

df = conversion_seconds_15
df.loc[(df['is_attributed'] != 0), 'threshold'] = df['click_time']/df['is_attributed']

df.plot(x ='hm_seconds_15', y='threshold', kind = 'line')
plt.savefig(OUTPUT_FOLDER + '/threshold_per15secs_clickSquared.png')
plt.clf()

co = conversion_seconds_15[conversion_seconds_15['conversion_rate']==0]
Q1 = co.click_time.quantile(0.25)
Q3 = co.click_time.quantile(0.75)
IQR = Q3 - Q1
click_outlier = Q3 + 1.5 * IQR
print("Number of clicks threshold: ", click_outlier)

Q1 = df.threshold.quantile(0.25)
Q3 = df.threshold.quantile(0.75)
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
print("Clicks squared per downloads threshold: ", threshold)

df.loc[(df['is_attributed'] == 0) & (df['click_time'] > click_outlier), 'isFraud'] = 1
df.loc[(df['is_attributed'] != 0) & (df['threshold'] > threshold), 'isFraud'] = 1
df.loc[(df['isFraud'].isnull()), 'isFraud'] = 0

new_data = pd.merge(train_2days, df[['hm_seconds_15', 'isFraud']], on='hm_seconds_15', how='left')
event_rate = new_data.isFraud.sum()/len(new_data)
print("Event rate per 15 seconds using click squared threshold: ", event_rate)
new_data.to_csv(OUTPUT_FOLDER + '/June7_TaggedData_per15secs.csv')

print("Done")
