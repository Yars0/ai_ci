import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np

df = pd.read_csv('dataset.csv')

import seaborn as sns

# Количество клиентов на одно устройство
device_clients = df.groupby('device_id')['client_id'].nunique()

# Построение тепловой карты
device_clients_df = df.groupby(['device_id', 'ip'])['client_id'].nunique().unstack(fill_value=0)

plt.figure(figsize=(12, 8))
sns.heatmap(device_clients_df, cmap='coolwarm', cbar=True)
plt.title('Использование устройств и IP разными клиентами')
plt.xlabel('IP')
plt.ylabel('ID устройства')
plt.show()

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN


# Кодируем IP и device_id в числовой формат для анализа
encoder_ip = LabelEncoder()
encoder_device = LabelEncoder()

df['ip_encoded'] = encoder_ip.fit_transform(df['ip'])
df['device_id_encoded'] = encoder_device.fit_transform(df['device_id'])

# Выбор данных для DBSCAN (2D: device_id_encoded и ip_encoded)
X = df[['device_id_encoded', 'ip_encoded']].values

# Применение DBSCAN для кластеризации
dbscan = DBSCAN(eps=12, min_samples=200)  # eps и min_samples можно регулировать
df['cluster'] = dbscan.fit_predict(X)

print('уникальные кластеры: ', df['cluster'].unique())
#plt.scatter(X[:,0], X[:,1], s=1)
#plt.show()

# Выделяем выбросы (cluster = -1)
anomalies = df[df['cluster'] == -1]



# Сводка о выбросах
anomalies_summary = anomalies.groupby(['device_id', 'ip']).size().reset_index(name='anomaly_count')
#tools.display_dataframe_to_user(name="Аномалии в использовании устройств и IP", dataframe=anomalies_summary)
print(anomalies_summary)

"""nearest_neighbors = NearestNeighbors(n_neighbors=50)
nearest_neighbors.fit(X)
distances, _ = nearest_neighbors.kneighbors(X)

# Отсортируем расстояния для визуализации
distances = np.sort(distances[:, 1])

# Построение графика расстояний
plt.figure(figsize=(12, 6))
plt.plot(distances)
plt.title('k-Distance Graph for DBSCAN (k=2)')
plt.xlabel('Points sorted by distance')
plt.ylabel('Distance to nearest neighbor')
plt.grid()
plt.show()"""




df['card_status'] = df['card_status'].replace({'blk': 'blocked', 'act': 'active'})
df['device_type'] = df['device_type'].str.lower().replace({
    'atm': 'atm',
    'portable term': 'portable_term',
    'prtbl trm': 'portable_term',
    'pos trm': 'pos_terminal',
    'port_trm': 'pos_terminal',
    'cash_in': 'cash_in',
    'cash_out': 'cash_out'
})

# Aggregate df for plotting
aggregated_data = df.groupby(['device_type', 'oper_type', 'card_status'])['pin_inc_count'].mean().reset_index()

# Pivot the df for a heatmap
pivot_data = aggregated_data.pivot_table(
    index='device_type', 
    columns='card_status', 
    values='pin_inc_count', 
    aggfunc='mean', 
    fill_value=0
)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Average PIN Incorrect Count by Device Type and Card Status")
plt.xlabel("Card Status")
plt.ylabel("Device Type")
plt.show()