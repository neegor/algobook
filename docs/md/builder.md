---
tags: [Патерны проектирования]
---

# Паттерн Builder (Строитель)

Паттерн Builder (Строитель) - это порождающий паттерн проектирования, который позволяет создавать сложные объекты пошагово. Он отделяет конструирование сложного объекта от его представления, так что в результате одного и того же процесса конструирования могут получаться разные представления.

## Когда использовать Builder?

- Когда объект имеет много параметров (некоторые из которых необязательные)
- Когда нужно создавать разные варианты одного объекта
- Когда процесс создания объекта должен быть независимым от его частей
- Когда нужно обеспечить контроль над процессом создания сложного объекта

## Основные компоненты Builder

1. **Product** - создаваемый сложный объект
2. **Builder** - абстрактный интерфейс для создания частей Product
3. **ConcreteBuilder** - конкретная реализация Builder, создает и собирает части Product
4. **Director** - отвечает за выполнение шагов построения (может отсутствовать в упрощенных реализациях)

## Пример 1: Классическая реализация Builder

```python
# Product - сложный объект, который мы строим
class Pizza:
    def __init__(self):
        self.dough = None
        self.sauce = None
        self.topping = None

    def __str__(self):
        return f"Пицца с тестом: {self.dough}, соусом: {self.sauce} и начинкой: {self.topping}"

# Abstract Builder
class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()

    def set_dough(self, dough):
        self.pizza.dough = dough

    def set_sauce(self, sauce):
        self.pizza.sauce = sauce

    def set_topping(self, topping):
        self.pizza.topping = topping

    def get_pizza(self):
        return self.pizza

# Concrete Builder
class MargheritaBuilder(PizzaBuilder):
    def build_dough(self):
        self.set_dough("тонкое")

    def build_sauce(self):
        self.set_sauce("томатный")

    def build_topping(self):
        self.set_topping("моцарелла и базилик")

# Director
class Waiter:
    def __init__(self):
        self.builder = None

    def construct_pizza(self, builder):
        self.builder = builder
        self.builder.build_dough()
        self.builder.build_sauce()
        self.builder.build_topping()

    def get_pizza(self):
        return self.builder.get_pizza()

# Использование
waiter = Waiter()
margherita_builder = MargheritaBuilder()
waiter.construct_pizza(margherita_builder)
pizza = waiter.get_pizza()
print(pizza)  # Пицца с тестом: тонкое, соусом: томатный и начинкой: моцарелла и базилик
```

## Пример 2: Упрощенный Builder (без Director)

```python
class Computer:
    def __init__(self):
        self.cpu = None
        self.ram = None
        self.storage = None

    def __str__(self):
        return f"Computer: CPU={self.cpu}, RAM={self.ram}, Storage={self.storage}"

class ComputerBuilder:
    def __init__(self):
        self.computer = Computer()

    def add_cpu(self, cpu):
        self.computer.cpu = cpu
        return self  # Возвращаем self для поддержки цепочки вызовов

    def add_ram(self, ram):
        self.computer.ram = ram
        return self

    def add_storage(self, storage):
        self.computer.storage = storage
        return self

    def build(self):
        return self.computer

# Использование
builder = ComputerBuilder()
computer = builder.add_cpu("Intel i7").add_ram("16GB").add_storage("512GB SSD").build()
print(computer)  # Computer: CPU=Intel i7, RAM=16GB, Storage=512GB SSD
```

## Пример 3: Builder с обязательными и необязательными параметрами

```python
class User:
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.phone = None
        self.address = None

    def __str__(self):
        return f"User: {self.username}, email: {self.email}, phone: {self.phone}, address: {self.address}"

class UserBuilder:
    def __init__(self, username, email):
        self.user = User(username, email)

    def set_phone(self, phone):
        self.user.phone = phone
        return self

    def set_address(self, address):
        self.user.address = address
        return self

    def build(self):
        return self.user

# Использование
user1 = UserBuilder("john_doe", "john@example.com").build()
print(user1)  # User: john_doe, email: john@example.com, phone: None, address: None

user2 = UserBuilder("jane_doe", "jane@example.com").set_phone("123456789").build()
print(user2)  # User: jane_doe, email: jane@example.com, phone: 123456789, address: None

user3 = UserBuilder("bob_smith", "bob@example.com").set_phone("987654321").set_address("123 Main St").build()
print(user3)  # User: bob_smith, email: bob@example.com, phone: 987654321, address: 123 Main St
```

## Преимущества Builder

- Позволяет изменять внутреннее представление продукта
- Изолирует код для создания и представления
- Дает более тонкий контроль над процессом конструирования
- Позволяет создавать объекты пошагово
- Упрощает создание объектов с большим количеством параметров

## Недостатки Builder

- Усложняет код из-за введения дополнительных классов
- Может быть избыточным для простых объектов

Builder особенно полезен, когда объект требует многошагового процесса создания или когда нужно создавать разные варианты одного объекта.
