import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("csv/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                     DF Describe')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print(df.describe())
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                         DF Info')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
df.info()
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                     Max Column Values')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Convert string value of 'Label' column to int
non_numeric_label = {"BENIGN": 0, "DDoS": 1}
df[" Label"] = df[" Label"].map(non_numeric_label)

# A list by column names
columns = df.columns.tolist()

# Maximum value of each column
for i in range(0, len(columns)):
    print(f"{columns[i]} -> {df[columns[i]].max()}")
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                     Most used port')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Put all ports and their usage in a dictonary
# key: port, value: usage
port_dicts = {}
for i in range(df.shape[0]):
    port_in_the_data = df[" Destination Port"][i]
    if port_in_the_data in port_dicts:
        port_dicts[port_in_the_data] += 1
    else:
        port_dicts.update({port_in_the_data: 1})

# Sort port usage pair with ascending order of port
port_dicts = OrderedDict(sorted(port_dicts.items()))

# Find most used port value
values = port_dicts.values()
max_used_port_value = max(values)
max_used_port = 0

# Find the port adjacent to max used value
for i in port_dicts:
    if port_dicts[i] == max_used_port_value:
        print(f"Most used port is '{i}' with '{max_used_port_value}' calls")
        max_used_port = i
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('             Attacked and Not attacked values for most used port')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Attacked and Not attacked values for most used port
print(f"Port {max_used_port}, used: {len(df.loc[(df[' Destination Port'] == 80)])}")
print(f"Port {max_used_port}, attacked: {len(df.loc[(df[' Destination Port'] == 80) & (df[' Label'] == 1)])}")
print(f"Port {max_used_port}, not attacked: {len(df.loc[(df[' Destination Port'] == 80) & (df[' Label'] == 0)])}")
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Optimize int type values
for i in columns:
    if df[f"{i}"].dtype == 'int64':
        if df[f"{i}"].max() < 127:
            df[f"{i}"] = df[f"{i}"].astype('int8')
        elif df[f"{i}"].max() < 32767:
            df[f"{i}"] = df[f"{i}"].astype('int16')
        elif df[f"{i}"].max() < 2 * 10 ** 9:
            df[f"{i}"] = df[f"{i}"].astype('int32')
    else:
        continue
print('                                 Null values in columns')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')

# Visualize how many null values exist in data in descending
print(df.isnull().sum(axis=0).sort_values(ascending=False))
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                 After dropping null values')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Drop null values(samples) and replace
df.dropna(subset=['Flow Bytes/s'], inplace=True)
print(df.isnull().sum(axis=0).sort_values(ascending=False))
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print('                                       DF Describe')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# pd.set_option('display.max_columns', None)
print(df.describe())
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
# Flow Bytes/s and Flow Packets/s max values are infinity
df = df.drop('Flow Bytes/s', axis='columns')
df = df.drop(' Flow Packets/s', axis='columns')
columns = df.columns.tolist()

# Except from ports column, normalize all features
scaler = MinMaxScaler(feature_range=(0, 1))
for i in columns:
    try:
        if i == ' Destination Port':
            continue
        else:
            df[[f"{i}"]] = scaler.fit_transform(df[[f"{i}"]])
    except ValueError:
        continue

# Delete columns with all zero values
for k in range(len(columns)):
    if df[columns[k]].min() == 0 and df[columns[k]].max() == 0:
        df = df.drop(columns[k], axis='columns')

print('                                     Data normalized')
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
print(df.describe())
print(df.shape)
print('--------x--------x--------x--------x--------x--------x--------x--------x--------x--------')
df.to_parquet('pre_processed_dataset.par')
