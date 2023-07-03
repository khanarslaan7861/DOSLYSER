import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)
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

# Find the port adjacent to max used value
for i in port_dicts:
    if port_dicts[i] == max_used_port_value:
        print(f"Most used port is '{i}' with '{max_used_port_value}' calls")
        max_used_port = i

