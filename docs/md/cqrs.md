---
tags: [Патерны проектирования]
---

# CQRS (Command Query Responsibility Segregation)

## Что такое CQRS?

**CQRS** (**Command Query Responsibility Segregation**) - это архитектурный шаблон, который разделяет операции чтения (`queries`) и операции записи (`commands`) в приложении. Основная идея заключается в том, что модели для чтения и записи могут быть разными, что позволяет оптимизировать каждую из них для своей задачи.

## Основные принципы CQRS

1. **Разделение команд и запросов**:

   - Команды (Commands) - изменяют состояние системы (CREATE, UPDATE, DELETE)
   - Запросы (Queries) - получают данные без изменения состояния (READ)

2. **Разные модели**:

   - Модель записи (Write Model) - оптимизирована для бизнес-логики и валидации
   - Модель чтения (Read Model) - оптимизирована для отображения данных

3. **Разные хранилища (опционально)**:
   - Для записи и чтения могут использоваться разные базы данных

## Преимущества CQRS

- Улучшение масштабируемости (чтение и запись можно масштабировать отдельно)
- Упрощение моделей (каждая модель решает только одну задачу)
- Гибкость в выборе хранилищ
- Улучшение производительности (оптимизация под конкретные сценарии)

## Недостатки CQRS

- Усложнение архитектуры
- Проблемы согласованности данных (eventual consistency)
- Дополнительные накладные расходы на синхронизацию моделей

## Пример реализации CQRS на Python

### 1. Базовый пример без разделения хранилищ

```python
class UserWriteModel:
    def __init__(self):
        self.users = {}

    def create_user(self, user_id, name, email):
        if user_id in self.users:
            raise ValueError("User already exists")
        # Сложная бизнес-логика валидации
        if not email or "@" not in email:
            raise ValueError("Invalid email")
        self.users[user_id] = {"name": name, "email": email}

    def update_user(self, user_id, name=None, email=None):
        if user_id not in self.users:
            raise ValueError("User not found")
        if email is not None:
            if not email or "@" not in email:
                raise ValueError("Invalid email")
            self.users[user_id]["email"] = email
        if name is not None:
            self.users[user_id]["name"] = name


class UserReadModel:
    def __init__(self, write_model):
        self.write_model = write_model

    def get_user(self, user_id):
        user = self.write_model.users.get(user_id)
        if not user:
            return None
        # Оптимизированное представление для чтения
        return {
            "user_id": user_id,
            "user_name": user["name"],
            "user_email": user["email"],
            "email_provider": user["email"].split("@")[-1]
        }

    def list_users(self):
        return [
            {"user_id": uid, "name": data["name"]}
            for uid, data in self.write_model.users.items()
        ]


# Использование
write_model = UserWriteModel()
read_model = UserReadModel(write_model)

# Команды (изменяют состояние)
write_model.create_user(1, "Alice", "alice@example.com")
write_model.create_user(2, "Bob", "bob@example.org")

# Запросы (получают данные)
print(read_model.get_user(1))  # {'user_id': 1, 'user_name': 'Alice', ...}
print(read_model.list_users())  # [{'user_id': 1, 'name': 'Alice'}, ...]
```

### 2. Пример с разделением хранилищ и событийной синхронизацией

```python
from typing import Dict, List
import json

# Модель записи
class UserWriteModel:
    def __init__(self, event_store):
        self.event_store = event_store

    def create_user(self, user_id, name, email):
        # Валидация
        if not email or "@" not in email:
            raise ValueError("Invalid email")

        # Генерируем событие
        event = {
            "type": "UserCreated",
            "data": {
                "user_id": user_id,
                "name": name,
                "email": email
            }
        }
        self.event_store.publish(event)

    def update_email(self, user_id, new_email):
        # В реальности здесь была бы проверка существования пользователя
        event = {
            "type": "UserEmailUpdated",
            "data": {
                "user_id": user_id,
                "new_email": new_email
            }
        }
        self.event_store.publish(event)


# Модель чтения
class UserReadModel:
    def __init__(self):
        self.users = {}
        self.email_provider_stats = {}

    def apply_event(self, event):
        event_type = event["type"]
        data = event["data"]

        if event_type == "UserCreated":
            user_id = data["user_id"]
            self.users[user_id] = {
                "name": data["name"],
                "email": data["email"]
            }
            provider = data["email"].split("@")[-1]
            self.email_provider_stats[provider] = self.email_provider_stats.get(provider, 0) + 1

        elif event_type == "UserEmailUpdated":
            user_id = data["user_id"]
            old_email = self.users[user_id]["email"]
            old_provider = old_email.split("@")[-1]
            new_provider = data["new_email"].split("@")[-1]

            # Обновляем пользователя
            self.users[user_id]["email"] = data["new_email"]

            # Обновляем статистику
            self.email_provider_stats[old_provider] -= 1
            if self.email_provider_stats[old_provider] == 0:
                del self.email_provider_stats[old_provider]
            self.email_provider_stats[new_provider] = self.email_provider_stats.get(new_provider, 0) + 1

    def get_user(self, user_id):
        user = self.users.get(user_id)
        if not user:
            return None
        return {
            "user_id": user_id,
            "name": user["name"],
            "email": user["email"],
            "email_provider": user["email"].split("@")[-1]
        }

    def get_email_providers_stats(self):
        return self.email_provider_stats


# Хранилище событий
class EventStore:
    def __init__(self):
        self.events = []
        self.subscribers = []

    def publish(self, event):
        self.events.append(event)
        for subscriber in self.subscribers:
            subscriber.apply_event(event)

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)
        # Воспроизводим все прошлые события для нового подписчика
        for event in self.events:
            subscriber.apply_event(event)


# Использование
event_store = EventStore()
write_model = UserWriteModel(event_store)
read_model = UserReadModel()
event_store.subscribe(read_model)

# Выполняем команды
write_model.create_user(1, "Alice", "alice@example.com")
write_model.create_user(2, "Bob", "bob@example.org")
write_model.create_user(3, "Charlie", "charlie@example.com")

# Обновляем email
write_model.update_email(1, "alice@newdomain.com")

# Выполняем запросы
print(read_model.get_user(1))
# {'user_id': 1, 'name': 'Alice', 'email': 'alice@newdomain.com', 'email_provider': 'newdomain.com'}

print(read_model.get_email_providers_stats())
# {'example.org': 1, 'example.com': 1, 'newdomain.com': 1}
```

### 3. Пример с использованием разных баз данных

```python
# Предположим, что у нас есть:
# - PostgreSQL для записи (реляционная модель)
# - MongoDB для чтения (документ-ориентированная модель)

import psycopg2
from pymongo import MongoClient
from datetime import datetime

# Настройка подключений
pg_conn = psycopg2.connect("dbname=write_db user=postgres")
mongo_client = MongoClient("mongodb://localhost:27017/")
read_db = mongo_client["read_db"]

class UserCommandHandler:
    def __init__(self, pg_conn, event_bus):
        self.pg_conn = pg_conn
        self.event_bus = event_bus

    def create_user(self, name, email):
        # Валидация
        if not email or "@" not in email:
            raise ValueError("Invalid email")

        # Запись в PostgreSQL
        with self.pg_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (name, email, created_at) VALUES (%s, %s, %s) RETURNING id",
                (name, email, datetime.utcnow())
            )
            user_id = cur.fetchone()[0]
            self.pg_conn.commit()

        # Публикуем событие
        self.event_bus.publish({
            "type": "UserCreated",
            "data": {
                "user_id": user_id,
                "name": name,
                "email": email,
                "created_at": datetime.utcnow().isoformat()
            }
        })

        return user_id


class UserReadModel:
    def __init__(self, mongo_db):
        self.users = mongo_db["users"]

    def handle_event(self, event):
        event_type = event["type"]
        data = event["data"]

        if event_type == "UserCreated":
            # Оптимизированное представление для чтения в MongoDB
            self.users.insert_one({
                "user_id": data["user_id"],
                "name": data["name"],
                "email": data["email"],
                "email_provider": data["email"].split("@")[-1],
                "created_at": data["created_at"],
                "search_terms": [data["name"].lower(), data["email"].lower()]
            })

    def get_user(self, user_id):
        return self.users.find_one({"user_id": user_id}, {"_id": 0})

    def search_users(self, term):
        term = term.lower()
        return list(self.users.find({
            "search_terms": term
        }, {"_id": 0}))


class EventBus:
    def __init__(self):
        self.subscribers = []

    def publish(self, event):
        for subscriber in self.subscribers:
            subscriber.handle_event(event)

    def subscribe(self, subscriber):
        self.subscribers.append(subscriber)


# Инициализация
event_bus = EventBus()
command_handler = UserCommandHandler(pg_conn, event_bus)
read_model = UserReadModel(read_db)
event_bus.subscribe(read_model)

# Создаем пользователя (Command)
user_id = command_handler.create_user("Alice", "alice@example.com")

# Читаем данные (Query)
print(read_model.get_user(user_id))
print(read_model.search_users("alice"))
```

## Когда использовать CQRS?

CQRS хорошо подходит для:

- Систем с высокой нагрузкой на чтение или запись
- Сложных доменных моделей
- Систем, где требования к чтению и записи сильно отличаются
- Систем, использующих event sourcing

## Когда не стоит использовать CQRS?

- Простые CRUD-приложения
- Когда eventual consistency неприемлема
- В небольших проектах, где сложность не оправдана

## Заключение

CQRS - это мощный шаблон, который может значительно улучшить архитектуру сложных систем, но он вводит дополнительную сложность. Решение о его использовании должно быть взвешенным и основываться на конкретных требованиях проекта.

Представленные примеры демонстрируют основные концепции CQRS, но в реальных проектах могут потребоваться дополнительные механизмы, такие как:

- Компенсационные транзакции
- Механизмы повторной обработки событий
- Более сложные стратегии синхронизации
- Оптимистичные блокировки для модели записи
