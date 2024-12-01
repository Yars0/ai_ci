import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


# Загрузка исходного файла
df = pd.read_csv('dataset.csv')

# Кодируем IP и device_id в числовой формат для анализа
encoder_ip = LabelEncoder()
encoder_device = LabelEncoder()

df['ip_encoded'] = encoder_ip.fit_transform(df['ip'])
df['device_id_encoded'] = encoder_device.fit_transform(df['device_id'])

# Выбор данных для DBSCAN (2D: device_id_encoded и ip_encoded)
X = df[['device_id_encoded', 'ip_encoded']].values

# 1. Определите признаки для модели
features = ['device_id', 'ip_encoded', 'pin_inc_count', 'tran_code']  # Включите значимые для анализа параметры
X = df[features].fillna(0)  # Заменяем пропуски на 0

# Масштабирование данных
#scaler = StandardScaler()
#X = scaler.fit_transform(df[features])

from joblib import load

# Загрузка модели
model = load('model.pkl')

# Получение предсказаний
predictions = model.predict(X)

# Преобразование предсказаний в True/False
predictions = predictions > 0.5  # Для классификатора, если это вероятности
# Создание DataFrame с предсказаниями
predictions_df = pd.DataFrame(predictions, columns=["Prediction"])

# Сохранение в CSV файл
predictions_df.to_csv('predict.csv', index=False, header=False)

print("Файл predict.csv успешно создан.")

