---
tags: [Патерны проектирования]
---

# Паттерн Proxy (Заместитель)

Паттерн Proxy (Заместитель) — это структурный паттерн проектирования, который предоставляет объект-заменитель или placeholder для другого объекта. Прокси контролирует доступ к оригинальному объекту, позволяя выполнять действия до или после обращения к нему.

## Основные цели Proxy:

- Контроль доступа к объекту
- Добавление дополнительной логики перед/после обращения к объекту
- Ленивая инициализация (отложенное создание объекта)
- Кеширование результатов
- Удалённый доступ (например, для объектов в другом адресном пространстве)

## Типы прокси:

- **Virtual Proxy** - откладывает создание ресурсоёмких объектов
- **Protection Proxy** - контролирует доступ к объекту
- **Remote Proxy** - представляет объект в другом адресном пространстве
- **Smart Reference** - добавляет дополнительную логику при обращении к объекту
- **Caching Proxy** - кеширует результаты запросов

## Примеры на Python

### 1. Virtual Proxy (Ленивая инициализация)

```python
class LazyImage:
    def __init__(self, filename):
        self._filename = filename
        self._image = None

    def display(self):
        if self._image is None:
            print(f"Loading image {self._filename}")
            self._image = f"Image data for {self._filename}"
        print(f"Displaying {self._filename}")
        return self._image

# Использование
image = LazyImage("photo.jpg")
# Изображение ещё не загружено
print(image.display())  # Загружается и отображается
print(image.display())  # Только отображается (уже загружено)
```

### 2. Protection Proxy (Контроль доступа)

```python
class SensitiveData:
    def __init__(self):
        self._data = "Top Secret Data"

    def read(self):
        return self._data

class DataProxy:
    def __init__(self, user):
        self._user = user
        self._real_data = SensitiveData()

    def read(self):
        if self._user == "admin":
            return self._real_data.read()
        else:
            return "Access Denied"

# Использование
admin_proxy = DataProxy("admin")
print(admin_proxy.read())  # "Top Secret Data"

user_proxy = DataProxy("user")
print(user_proxy.read())  # "Access Denied"
```

### 3. Remote Proxy (Удалённый доступ)

```python
import json
from abc import ABC, abstractmethod

class DatabaseService(ABC):
    @abstractmethod
    def get_data(self, query):
        pass

class RealDatabaseService(DatabaseService):
    def get_data(self, query):
        # В реальности здесь было бы подключение к БД
        return {"result": f"Data for query: {query}"}

class DatabaseProxy(DatabaseService):
    def __init__(self):
        self._real_service = None
        self._cache = {}

    def get_data(self, query):
        # Ленивая инициализация
        if self._real_service is None:
            print("Connecting to remote database...")
            self._real_service = RealDatabaseService()

        # Кеширование
        if query in self._cache:
            print("Returning cached result")
            return self._cache[query]

        result = self._real_service.get_data(query)
        self._cache[query] = result
        return result

# Использование
proxy = DatabaseProxy()
print(proxy.get_data("SELECT * FROM users"))  # Соединение + запрос
print(proxy.get_data("SELECT * FROM users"))  # Возврат из кеша
```

### 4. Smart Proxy (Дополнительная логика)

```python
class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance

    def deposit(self, amount):
        self._balance += amount

    def withdraw(self, amount):
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount

    def get_balance(self):
        return self._balance

class BankAccountProxy:
    def __init__(self, real_account, owner):
        self._real_account = real_account
        self._owner = owner
        self._access_count = 0

    def deposit(self, amount):
        self._access_count += 1
        print(f"Log: {self._owner} deposited {amount}")
        self._real_account.deposit(amount)

    def withdraw(self, amount):
        self._access_count += 1
        print(f"Log: {self._owner} tried to withdraw {amount}")
        self._real_account.withdraw(amount)

    def get_balance(self):
        self._access_count += 1
        print(f"Log: {self._owner} checked balance")
        return self._real_account.get_balance()

    def get_access_count(self):
        return self._access_count

# Использование
account = BankAccount(100)
proxy = BankAccountProxy(account, "John Doe")

proxy.deposit(50)
proxy.withdraw(30)
print(f"Balance: {proxy.get_balance()}")
print(f"Access count: {proxy.get_access_count()}")
```

## Преимущества Proxy:

- Контроль доступа к реальному объекту
- Дополнительные возможности без изменения реального объекта
- Ленивая инициализация ресурсоёмких объектов
- Кеширование результатов
- Упрощение работы с удалёнными объектами

## Недостатки:

- Увеличение времени отклика из-за дополнительной логики
- Усложнение кода (введение дополнительных классов)

Паттерн Proxy особенно полезен, когда нужно добавить дополнительное поведение к объекту без изменения его кода или когда создание реального объекта является ресурсоёмкой операцией.
