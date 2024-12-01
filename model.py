from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from joblib import dump

# Имена столбцов (взяты из описания набора данных)
column_names = [
'idx',
'transaction_id',
'ip', 
 'device_id',
 'device_type', 
 'tran_code',
 'mcc',
 'client_id',
 'card_type',
 'pin_inc_count',
 'card_status',
 'expiration_date',
 'datetime',
 'sum',
 'oper_type',
 'oper_status',
 'balance'
]

# Загрузка данных

df = pd.read_csv('dataset.csv')

df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')


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
df['label'] = df['cluster'].apply(lambda x: 1 if x == -1 else 0)

# 1. Определите признаки для модели
features = ['device_id', 'ip_encoded', 'pin_inc_count', 'tran_code']  # Включите значимые для анализа параметры
X = df[features].fillna(0)  # Заменяем пропуски на 0
y = df['label']  # Целевой признак

# 2. Разделите данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 3. Выберите модель (например, Random Forest)
model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)

# 4. Обучите модель
model.fit(X_train, y_train)

# 5. Сделайте прогнозы
y_pred = model.predict(X_test)

# 6. Оцените качество модели
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. (Опционально) Важность признаков
importances = model.feature_importances_
for feature, importance in zip(features, importances):
    print(f"{feature}: {importance:.4f}")


"""prediction = model.predict(y_train.add())
file_name = 'predict.csv'
with open(file_name, 'w') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows()"""


dump(model, 'model_v2.pkl')  # Сохранение модели в файл
