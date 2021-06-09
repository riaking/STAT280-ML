
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
train['minute'] = train['minute'].apply(lambda x: '{0:0>2}'.format(x))
train['hour_minute'] = train['hour'].astype(str) + train['minute'].astype(str)
train['hour_minute'] = train['hour_minute'].astype('int32')
train.head()

conversion_minute = train.groupby('hour_minute').aggregate(
                        {
                            'is_attributed':sum,
                            'click_time':"count"
                        }
                    )

conversion_minute['conversion_rate'] = conversion_minute['is_attributed']/conversion_minute['click_time']
conversion_minute.reset_index(level=0, inplace=True)
conversion_minute.head()

conversion_minute.plot(x ='hour_minute', y='conversion_rate', kind = 'line')
plt.savefig(OUTPUT_FOLDER + '/conversionRate_hourPerMinute.png')
plt.clf()

df = conversion_minute
df.loc[(df['is_attributed'] != 0), 'threshold'] = df['click_time']/df['conversion_rate']

df.plot(x ='hour_minute', y='threshold', kind = 'line')
plt.savefig(OUTPUT_FOLDER + '/threshold_hourPerMinute_clickSquared.png')
plt.clf()

Q1 = df.threshold.quantile(0.25)
Q3 = df.threshold.quantile(0.75)
IQR = Q3 - Q1
threshold = Q3 + 1.5 * IQR
print("Clicks squared per downloads threshold: ", threshold)

df.loc[(df['is_attributed'] != 0) & (df['threshold'] > threshold), 'isFraud'] = 1
df.loc[(df['isFraud'].isnull()), 'isFraud'] = 0

new_data = pd.merge(train, df[['hour_minute', 'isFraud']], on='hour_minute', how='left')
event_rate = new_data.isFraud.sum()/len(new_data)
print("Event rate for hour per minute using click squared threshold: ", event_rate)
new_data.to_csv('TaggedData_HourPerMinute.csv')

print("Done")
