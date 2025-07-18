---
tags: [Патерны проектирования]
---

# Паттерн Prototype (Прототип)

Паттерн Prototype (Прототип) — это порождающий паттерн проектирования, который позволяет создавать новые объекты путем копирования существующих (прототипов), вместо создания объектов через конструктор. Это особенно полезно, когда создание объекта требует больших затрат ресурсов или когда система должна быть независимой от того, как создаются, компонуются и представляются её продукты.

## Когда использовать Prototype

1. Когда нужно избежать построения иерархий фабрик или конструкторов
2. Когда создание объекта дороже, чем его копирование
3. Когда классы создаются во время выполнения
4. Когда нужно легко добавлять и удалять объекты во время выполнения

## Реализация в Python

В Python прототипный паттерн можно реализовать с помощью встроенного модуля `copy`, который предоставляет функции `copy()` (поверхностное копирование) и `deepcopy()` (глубокое копирование).

### Базовый пример

```python
import copy

class Prototype:
    def __init__(self):
        self._objects = {}

    def register_object(self, name, obj):
        """Регистрирует объект для клонирования"""
        self._objects[name] = obj

    def unregister_object(self, name):
        """Удаляет объект из регистрации"""
        del self._objects[name]

    def clone(self, name, **attrs):
        """Клонирует зарегистрированный объект и обновляет его атрибуты"""
        obj = copy.deepcopy(self._objects[name])
        obj.__dict__.update(attrs)
        return obj

class Car:
    def __init__(self):
        self.make = "Toyota"
        self.model = "Camry"
        self.color = "Silver"

    def __str__(self):
        return f"{self.color} {self.make} {self.model}"

# Использование
prototype = Prototype()
car = Car()
prototype.register_object("basic_car", car)

# Клонирование базового автомобиля
car1 = prototype.clone("basic_car")
print(car1)  # Silver Toyota Camry

# Клонирование с изменением свойств
car2 = prototype.clone("basic_car", color="Red", model="Corolla")
print(car2)  # Red Toyota Corolla
```

### Пример с графическими объектами

```python
import copy

class Shape:
    def __init__(self):
        self.id = None
        self.type = None

    def clone(self):
        return copy.deepcopy(self)

    def get_type(self):
        return self.type

    def get_id(self):
        return self.id

    def set_id(self, id):
        self.id = id

class Rectangle(Shape):
    def __init__(self):
        super().__init__()
        self.type = "Rectangle"

class Circle(Shape):
    def __init__(self):
        super().__init__()
        self.type = "Circle"

class ShapeCache:
    _shape_map = {}

    @staticmethod
    def get_shape(shape_id):
        cached_shape = ShapeCache._shape_map.get(shape_id, None)
        return cached_shape.clone() if cached_shape else None

    @staticmethod
    def load():
        circle = Circle()
        circle.set_id("1")
        ShapeCache._shape_map[circle.get_id()] = circle

        rectangle = Rectangle()
        rectangle.set_id("2")
        ShapeCache._shape_map[rectangle.get_id()] = rectangle

# Использование
ShapeCache.load()

circle = ShapeCache.get_shape("1")
print(f"Shape: {circle.get_type()}")  # Shape: Circle

rectangle = ShapeCache.get_shape("2")
print(f"Shape: {rectangle.get_type()}")  # Shape: Rectangle
```

### Пример с персонажами игры

```python
import copy

class GameCharacter:
    def __init__(self, health, speed, attack_power):
        self.health = health
        self.speed = speed
        self.attack_power = attack_power

    def clone(self):
        return copy.deepcopy(self)

    def __str__(self):
        return f"Health: {self.health}, Speed: {self.speed}, Attack: {self.attack_power}"

# Создаем прототипы персонажей
warrior_prototype = GameCharacter(100, 50, 80)
mage_prototype = GameCharacter(60, 30, 120)
archer_prototype = GameCharacter(80, 70, 60)

# Клонируем персонажей
warrior1 = warrior_prototype.clone()
warrior2 = warrior_prototype.clone()
warrior2.health = 120  # Модифицируем клон

mage1 = mage_prototype.clone()
archer1 = archer_prototype.clone()

print(warrior1)  # Health: 100, Speed: 50, Attack: 80
print(warrior2)  # Health: 120, Speed: 50, Attack: 80
print(mage1)     # Health: 60, Speed: 30, Attack: 120
print(archer1)   # Health: 80, Speed: 70, Attack: 60
```

## Преимущества и недостатки

**Преимущества:**

- Позволяет добавлять и удалять объекты во время выполнения
- Скрывает сложность создания новых объектов
- Уменьшает количество подклассов
- Позволяет динамически конфигурировать приложение классами

**Недостатки:**

- Сложность реализации, если объекты имеют циклические ссылки
- Необходимость реализации механизма клонирования для каждого класса

Prototype особенно полезен в Python благодаря встроенной поддержке копирования объектов, что делает его реализацию более простой по сравнению с некоторыми другими языками.
