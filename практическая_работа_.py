# -*- coding: utf-8 -*-

"""# Новый раздел

ЗАДАНИЕ 1
"""

import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix

df=pd.read_csv('/fake_news.csv')
df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

# Разделение на тренировочный и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    stop_words='english',
    max_df=0.7,
    ngram_range=(1, 2)  # Учитываем униграммы и биграммы
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = PassiveAggressiveClassifier(
    max_iter = 1000,
    random_state = 42,
    early_stopping = True
)
model.fit(X_train_tfidf,y_train)



import seaborn as sns
import matplotlib.pyplot as plt

#Предсказания и точность
y_pred = model.predict(X_test_tfidf)
accurasy = accuracy_score(y_test,y_pred)
print(f'Точность модели:{accurasy:.2%}')

# Матрица ошибок
cm = confusion_matrix(y_test, y_pred, normalize='true')
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['REAL', 'FAKE'],
            yticklabels=['REAL', 'FAKE'])
plt.title('Confusion Matrix')
plt.ylabel('Истинные значения')
plt.xlabel('Предсказанные значения')
plt.show()

df.shape

df.isnull().sum()

df.label

df.label.value_counts()

i=df.label.value_counts()

fig = go.Figure(data=[go.Bar(
            x=['Real','Fake'], y=i,
            text=i,
            textposition='auto',
        )])

fig.show()

df.isnull()

df.tail()

df.head()

"""2 **ЗАДАНИЕ**"""

# Шаг 1: Импорт всех необходимых библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Шаг 2: Загрузка данных
# Загружаем данные из файла (предполагается, что файл 'parkinsons.data' находится в той же директории)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
data = pd.read_csv(url)

#Первичный осмотр данных
print("Размер датасета:", data.shape)
print("n\Первые 5 строк:")
print(data.head())
print("\nИнформация о данных:")
print(data.info())
print("\nПроверка пропущенных значений:")
print(data.isnull().sum())

# Важный момент: столбец 'name' является уникальным идентификатором, а не признаком.
# Его необходимо удалить, чтобы модель не подстраивалась под него.
data = data.drop(columns=['name'], axis = 1)
X = data.drop(columns=['status'],axis = 1)
y = data['status']

#Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size= 0.2, random_state=42,stratify=y)
#Проверяем размеры выборок
print(f"\nРазмер обучающей выборки:{X_train.shape}")
print(f"Размер тестовой выборки:{X_test.shape}")

import xgboost as xgb
#Создание и обучение модели XGboost
model = xgb.XGBClassifier(
    objective = 'binary:logistic',
    random_state = 42,
    eval_metric = 'logloss',
    use_label_encoder = False
)

#Обучаем модель на тренировочных данных
model.fit(X_train, y_train)

#Предсказание и оценка точности
y_pred = model.predict(X_test)

#Рассчитываем точность
accuracy = accuracy_score(y_test,y_pred)
print(f"\nТочность модели на тестовой выборке: {accuracy:.4f}")

#Выводим подробный отчет по метрикам классификации
print("\nОтчет о классификации:")
print(classification_report(y_test,y_pred))

#Выводим матрицу ошибок
print("\nМатрица ошибок(Confusion Matrix):")
print(confusion_matrix(y_test,y_pred))

#Анализ важности признаков
import matplotlib.pyplot as plt

#Создаем датафрейм для визуализации важности признаков
feature_importances = pd.DataFrame({
    'feature':data.drop(columns=['status']).columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

#Визуализируем топ 10 важных признаков
plt.figure(figsize=(10,6))
plt.barh(feature_importances['feature'][:10],feature_importances['importance'][:10])
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков')
plt.gca().invert_yaxis()
plt.show

#Настройка гиперпараметров для повышения точности >95%
#Используем GridSearchCV для поиска лучших параметров


#Определяем сетку параметров для перебора
param_grid = {
    'max_depth':[3,4,5,6],
    'subsample': [0.01,0.1,0.2],
    'colsample_bytree': [0.8,0.9,1.0],
    'gamma': [0,0.1,0.2],
    'n_estimators': [100,200,300]

}

#Создаем модель для поиска
grid_model = xgb.XGBClassifier(objective='binary:logistic',random_state = 42,use_label_encoder=False)

#Инициализируем GridSearchCV
grid_search = GridSearchCV(
    estimator = grid_model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv = 5,
    n_jobs =-1,
    verbose = 1

)
#Запускаем поиск по сетке
print("\nЗапускаем настройки гиперпараметров...")
grid_search.fit(X_train,y_train)

#Выводим лучшие параметры
print(f"\nЛучшие параметры: {grid_search.best_params_}")

#Создаем итоговую модель с лучшими найденными параметрами
best_model = grid_search.best_estimator_

#Оцениваем ее производительность на тестовых данных
y_pred_best = best_model.predict(X_test)
best_accuracy = accuracy_score(y_test,y_pred_best)
print(f"\nТочность настроенной модели на тестовой выборке : {best_accuracy:.4f}")

#Сравниваем с базовой моделью
if best_accuracy > accuracy:
  print("Настройка гиперпараметров позволила улучшить точность модели")
else:
  print("Базовая модель показала лучший результат")

"""Задание номер **3**"""

Творческая работа. Я решил сделать модель  с использованием LightGBM и сделать модель которая прогнозирует примерную мощность двигателя для определенного типа автомомбиля Ссылка на датасет https://www.kaggle.com/datasets/CooperUnion/cardataset?resource=download

!pip install pandas matplotlib seaborn

!pip install catboost

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

df=pd.read_csv('/content/data.csv.zip')

y = df["Engine HP"]

X = df.drop("Engine HP",axis=1)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMRegressor(n_estimators=500,learning_rate=0.05,random_state=42)
xgb_model = xgb.XGBRegressor(n_estimators=500,learning_rate=0.05,random_state=42)
cat_model = CatBoostRegressor(n_estimators=500,learning_rate=0.05,depth=8, verbose = 0,random_state=42)

#Очистка данных
X.columns = (
    X.columns
    .str.replace('[^A-Za-z0-9_]+', '_', regex=True)  # заменим спецсимволы на "_"
    .str.strip('_')  # уберём лишние подчеркивания по краям
)

# Убираем строки с NaN или бесконечностями в y
mask = (~y.isna()) & np.isfinite(y)
X = X.loc[mask]
y = y.loc[mask]

# На всякий случай проверим X тоже
mask_X = np.all(np.isfinite(X), axis=1)
X = X.loc[mask_X]
y = y.loc[mask_X]

import re

def clean_column(col):
    return re.sub(r'\W+', '_', str(col))  # заменяет все не-буквенно-цифровые символы на "_"

X_train = X_train.rename(columns=clean_column)
X_test  = X_test.rename(columns=clean_column)

#Обучение модели
lgb_model.fit(X_train, y_train)

#Предсказание мощности
y_pred = lgb_model.predict(X_test)

df['Make'] = df['Make'].str.strip().str.title()

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns

# Группируем по бренду и берём медианное значение мощности
top10_Make = Make_median.sort_values(by='Engine HP', ascending=False).head(10)

# Берём топ-10 брендов
top10_Make = median_hp.head(10).reset_index()

# Строим график
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_Make, x='Make', y='pred_Engine HP', palette='viridis')

plt.title('Топ-10 марок по предсказанной мощности')
plt.xlabel('Марка')
plt.ylabel('Медианная предсказанная мощность (л.с.)')
plt.xticks(rotation=30)
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Берём медианные значения по брендам
median_hp = df.groupby('Make')['pred_Engine HP'].median().sort_values(ascending=False)

# Топ-10 брендов
top10_Make = median_hp.head(10).reset_index()

# Строим график
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_Make, x='Make', y='pred_Engine HP', palette='viridis')

plt.title('Топ-10 марок по предсказанной мощности')
plt.xlabel('Марка')
plt.ylabel('Медианная предсказанная мощность (л.с.)')
plt.xticks(rotation=30)
plt.show()

manufacturer_median = df.groupby('Make')['Engine HP'].median().reset_index()
top10_manufacturers = manufacturer_median.sort_values(by='Engine HP', ascending=False).head(10)

# 6. Построение графика
plt.figure(figsize=(10, 6))
sns.barplot(data=top10_manufacturers, x='Engine HP', y='Make', palette='viridis')
plt.title('Топ-10 марок по мощности')
plt.xlabel('Медианная мощность (л.с.)')
plt.ylabel('Марка')
plt.show()
