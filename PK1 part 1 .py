import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


np.random.seed(0)
data = {
    'feature1': np.random.randint(1, 100, 100),
    'feature2': np.random.randint(1, 100, 100),
    'feature3': np.random.rand(100) * 100
}
df = pd.DataFrame(data)


feature_to_scale = 'feature1'


scaler = StandardScaler()
df[feature_to_scale + '_scaled'] = scaler.fit_transform(df[[feature_to_scale]])

print(df.head())

from sklearn.feature_selection import SelectKBest, mutual_info_regression

# Целевая переменная (если имеется)
target_variable = None

# Выбираем количество лучших признаков
k_best_features = 10

# Удаляем целевую переменную (если имеется) из данных
if target_variable is not None:
    X = df.drop(columns=[target_variable])
else:
    X = df

# Определяем целевую переменную (если имеется)
if target_variable is not None:
    y = df[target_variable]

# Используем метод взаимной информации для отбора признаков
selector = SelectKBest(score_func=mutual_info_regression, k=k_best_features)
selected_features = selector.fit_transform(X, y)

# Получаем индексы отобранных признаков
selected_indices = selector.get_support(indices=True)

# Получаем названия отобранных признаков
selected_feature_names = X.columns[selected_indices]

# Выводим отобранные признаки
print("Отобранные признаки:")
print(selected_feature_names)
