---
tags:
  [
    Обработка текста,
    Компьютерное зрение,
    Экономика и финансы,
    Биотехнологии и медицина,
    Предиктивная аналитика,
  ]
---

# Преобразования данных в машинном обучении

Преобразования данных - это критически важный этап подготовки данных перед построением моделей машинного обучения. Они помогают улучшить качество данных, привести их к виду, который лучше подходит для алгоритмов, и часто значительно повышают производительность моделей.

## Основные бласти применения

1. Обработка текста — векторизация текста, лемматизация, токенизация  
2. Компьютерное зрение — нормализация изображений, аугментация данных, преобразование цветовых пространств  
3. Экономика и финансы — масштабирование признаков, обработка выбросов, создание производных показателей  
4. Биотехнологии и медицина — стандартизация данных, кодирование категориальных признаков, заполнение пропусков  
5. Предиктивная аналитика — полиномиальные преобразования, создание временных лагов, бинирование непрерывных переменных 

## Основные типы преобразований данных

### 1. Масштабирование и нормализация

#### Стандартизация (Z-score normalization)
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

data = np.array([[1, 2], [3, 4], [5, 6]])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

print("Исходные данные:\n", data)
print("Стандартизированные данные:\n", scaled_data)
print("Средние значения:", scaler.mean_)
print("Стандартные отклонения:", scaler.scale_)
```

#### Min-Max нормализация
```python
from sklearn.preprocessing import MinMaxScaler

data = np.array([[1, 2], [3, 4], [5, 6]])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

print("Исходные данные:\n", data)
print("Нормализованные данные:\n", scaled_data)
print("Минимальные значения:", scaler.data_min_)
print("Максимальные значения:", scaler.data_max_)
```

### 2. Кодирование категориальных признаков

#### One-Hot Encoding
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

data = pd.DataFrame({'color': ['red', 'blue', 'green', 'blue', 'red']})
encoder = OneHotEncoder(sparse=False)
encoded_data = encoder.fit_transform(data)

print("Исходные данные:\n", data)
print("Закодированные данные:\n", encoded_data)
print("Категории:", encoder.categories_)
```

#### Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

data = ['cat', 'dog', 'bird', 'dog', 'cat']
encoder = LabelEncoder()
encoded_data = encoder.fit_transform(data)

print("Исходные данные:", data)
print("Закодированные данные:", encoded_data)
print("Соответствие классов:", list(encoder.classes_))
```

### 3. Преобразование текстовых данных

#### TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'this is the first document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document'
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print("Словарь:", vectorizer.get_feature_names_out())
print("TF-IDF матрица:\n", X.toarray())
```

### 4. Обработка пропущенных значений

```python
from sklearn.impute import SimpleImputer
import numpy as np

data = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9]])
imputer = SimpleImputer(strategy='mean')
imputed_data = imputer.fit_transform(data)

print("Исходные данные:\n", data)
print("Данные после обработки пропусков:\n", imputed_data)
print("Значения для замены:", imputer.statistics_)
```

### 5. Преобразование распределения (нормализация)

#### Логарифмическое преобразование
```python
import numpy as np
from sklearn.preprocessing import FunctionTransformer

data = np.array([1, 10, 100, 1000]).reshape(-1, 1)
transformer = FunctionTransformer(np.log1p)
transformed_data = transformer.transform(data)

print("Исходные данные:", data.flatten())
print("После логарифмирования:", transformed_data.flatten())
```

#### Преобразование Бокса-Кокса
```python
from sklearn.preprocessing import PowerTransformer

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
transformer = PowerTransformer(method='box-cox', standardize=False)
transformed_data = transformer.fit_transform(data)

print("Исходные данные:", data.flatten())
print("После преобразования Бокса-Кокса:", transformed_data.flatten())
print("Лямбда параметр:", transformer.lambdas_)
```

### 6. Создание полиномиальных признаков
```python
from sklearn.preprocessing import PolynomialFeatures

data = np.array([[1, 2], [3, 4]])
poly = PolynomialFeatures(degree=2)
poly_data = poly.fit_transform(data)

print("Исходные данные:\n", data)
print("Полиномиальные признаки:\n", poly_data)
print("Имена признаков:", poly.get_feature_names_out())
```

### 7. Дискретизация (биннинг)
```python
from sklearn.preprocessing import KBinsDiscretizer

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
est.fit(data)
discretized = est.transform(data)

print("Исходные данные:", data.flatten())
print("После дискретизации:", discretized.flatten())
print("Границы бинов:", est.bin_edges_)
```

## Пайплайн преобразований

На практике часто используют несколько преобразований последовательно:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Пример данных
data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'salary': [50000, 60000, 70000, np.nan, 90000],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
})

# Определяем преобразования для разных типов признаков
numeric_features = ['age', 'salary']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['department']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Объединяем преобразования
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Применяем преобразования
processed_data = preprocessor.fit_transform(data)
print("Преобразованные данные:\n", processed_data)
```

## Важность преобразований данных

1. **Улучшение производительности моделей**: Многие алгоритмы (например, SVM, k-NN, нейронные сети) чувствительны к масштабу данных.

2. **Ускорение обучения**: Нормализованные данные часто позволяют алгоритмам сходиться быстрее.

3. **Интерпретируемость**: Преобразования могут сделать данные более понятными для анализа.

4. **Обработка выбросов**: Некоторые преобразования (например, логарифмирование) уменьшают влияние выбросов.

5. **Подготовка к специфичным алгоритмам**: Например, PCA требует масштабированных данных.

Выбор конкретных преобразований зависит от природы данных, выбранной модели и поставленной задачи. Часто пробуют несколько вариантов и выбирают тот, который дает лучший результат.