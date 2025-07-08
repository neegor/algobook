---
tags: [Патерны проектирования]
---

# Паттерн Singleton (Одиночка)

Singleton (Одиночка) - это порождающий паттерн проектирования, который гарантирует, что у класса есть только один экземпляр, и предоставляет глобальную точку доступа к этому экземпляру.

## Основные характеристики Singleton

1. **Единственный экземпляр**: Класс имеет только один экземпляр
2. **Глобальный доступ**: Экземпляр доступен из любой точки программы
3. **Ленивая инициализация**: Экземпляр создается только при первом обращении
4. **Потокобезопасность**: В многопоточной среде гарантирует создание только одного экземпляра

## Когда использовать Singleton?

Паттерн Singleton наиболее эффективен в следующих случаях:

1. **Управление общими ресурсами**:

   - Логгирование
   - Подключения к базам данных
   - Файловые системы

2. **Конфигурационные объекты**:

   - Настройки приложения
   - Параметры среды выполнения

3. **Кеширование**:

   - Общий кеш данных
   - Хранилище сессий

4. **Управление аппаратными ресурсами**:
   - Принтеры
   - Графические процессоры

## Реализация Singleton в Python

### 1. Базовый вариант (не потокобезопасный)

```python
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Использование
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # Выведет: True
```

### 2. Потокобезопасный вариант

```python
from threading import Lock

class ThreadSafeSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

# Использование
ts1 = ThreadSafeSingleton()
ts2 = ThreadSafeSingleton()
print(ts1 is ts2)  # Выведет: True
```

### 3. Singleton как декоратор

```python
def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Создано подключение к БД")

# Использование
db1 = DatabaseConnection()
db2 = DatabaseConnection()
print(db1 is db2)  # Выведет: True
```

### 4. Singleton через метакласс

```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=SingletonMeta):
    def __init__(self):
        self.logs = []

    def add_log(self, message):
        self.logs.append(message)

# Использование
logger1 = Logger()
logger2 = Logger()
logger1.add_log("Первое сообщение")
print(logger2.logs)  # Выведет: ['Первое сообщение']
print(logger1 is logger2)  # Выведет: True
```

## Практические примеры использования

### 1. Логгер

```python
class Logger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.log_file = open('app.log', 'a')
        return cls._instance

    def log(self, message):
        self.log_file.write(f"{message}\n")
        self.log_file.flush()

    def __del__(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()

# Использование
logger = Logger()
logger.log("Запуск приложения")
```

### 2. Конфигурация приложения

```python
class AppConfig:
    _instance = None
    _config = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Загрузка конфигурации из файла
            cls._instance._config = {
                'database': 'localhost:5432',
                'debug': True,
                'timeout': 30
            }
        return cls._instance

    def get(self, key):
        return self._config.get(key)

# Использование
config = AppConfig()
print(config.get('database'))  # Выведет: localhost:5432
```

### 3. Подключение к базе данных

```python
import sqlite3

class DatabaseConnection:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.connection = sqlite3.connect('mydatabase.db')
        return cls._instance

    def execute(self, query):
        cursor = self.connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()

# Использование
db = DatabaseConnection()
results = db.execute("SELECT * FROM users")
```

## Преимущества и недостатки Singleton

**Преимущества**:

- Контролируемый доступ к единственному экземпляру
- Глобальный доступ без использования глобальных переменных
- Ленивая инициализация (объект создается только при необходимости)

**Недостатки**:

- Нарушает принцип единственной ответственности (решает две задачи: управление своим жизненным циклом и основную функциональность)
- Усложняет тестирование (глобальное состояние)
- Может скрывать плохой дизайн (когда компоненты знают слишком много друг о друге)
- Проблемы в многопоточных средах (если не реализован правильно)

## Альтернативы Singleton

1. **Зависимости через конструктор** (Dependency Injection)
2. **Монады** (в функциональном программировании)
3. **Контекст приложения** (передавать нужные объекты явно)
