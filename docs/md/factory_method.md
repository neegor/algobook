---
tags: [Патерны проектирования]
---

# Паттерн Factory Method (Фабричный метод)

**Фабричный метод** — это порождающий паттерн проектирования, который определяет интерфейс для создания объектов, но позволяет подклассам изменять тип создаваемых объектов.

### Основная идея

- Класс делегирует создание объектов своим подклассам.
- Избавляет клиентский код от привязки к конкретным классам.
- Позволяет гибко управлять созданием объектов.

## Структура паттерна

1. **Абстрактный создатель (`Creator`)**

   - Объявляет фабричный метод (`factory_method()`), который возвращает объект продукта.
   - Может содержать базовую бизнес-логику, использующую продукты.

2. **Конкретные создатели (`ConcreteCreator`)**

   - Переопределяют фабричный метод и возвращают конкретные продукты.

3. **Абстрактный продукт (`Product`)**

   - Определяет интерфейс объектов, создаваемых фабричным методом.

4. **Конкретные продукты (`ConcreteProduct`)**
   - Реализуют интерфейс абстрактного продукта.

## Пример 1: Простая фабрика транспорта

```python
from abc import ABC, abstractmethod

# Абстрактный продукт
class Transport(ABC):
    @abstractmethod
    def deliver(self):
        pass

# Конкретные продукты
class Truck(Transport):
    def deliver(self):
        return "Доставка груза по земле"

class Ship(Transport):
    def deliver(self):
        return "Доставка груза по морю"

# Абстрактный создатель
class Logistics(ABC):
    @abstractmethod
    def create_transport(self) -> Transport:
        pass

    def plan_delivery(self) -> str:
        transport = self.create_transport()
        return f"Логистика: {transport.deliver()}"

# Конкретные создатели
class RoadLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Truck()

class SeaLogistics(Logistics):
    def create_transport(self) -> Transport:
        return Ship()

# Клиентский код
def client_code(logistics: Logistics):
    print(logistics.plan_delivery())

if __name__ == "__main__":
    client_code(RoadLogistics())  # Логистика: Доставка груза по земле
    client_code(SeaLogistics())   # Логистика: Доставка груза по морю
```

## Пример 2: Фабрика документов (Word, PDF)

```python
from abc import ABC, abstractmethod

# Абстрактный продукт
class Document(ABC):
    @abstractmethod
    def save(self):
        pass

# Конкретные продукты
class PDFDocument(Document):
    def save(self):
        return "Сохранение документа в формате PDF"

class WordDocument(Document):
    def save(self):
        return "Сохранение документа в формате Word"

# Абстрактный создатель
class Application(ABC):
    @abstractmethod
    def create_document(self) -> Document:
        pass

    def save_document(self):
        doc = self.create_document()
        return doc.save()

# Конкретные создатели
class PDFApplication(Application):
    def create_document(self) -> Document:
        return PDFDocument()

class WordApplication(Application):
    def create_document(self) -> Document:
        return WordDocument()

# Клиентский код
app1 = PDFApplication()
print(app1.save_document())  # Сохранение документа в формате PDF

app2 = WordApplication()
print(app2.save_document())  # Сохранение документа в формате Word
```

## Преимущества Factory Method

- Избавляет класс от привязки к конкретным классам продуктов.
- Выделяет код производства продуктов в отдельное место.
- Упрощает добавление новых продуктов.

## Недостатки

- Может привести к созданию большого числа подклассов.
