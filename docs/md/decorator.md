---
tags: [Патерны проектирования]
---

# Паттерн Decorator (Декоратор)

Паттерн Decorator (Декоратор) — это структурный паттерн проектирования, который позволяет динамически добавлять объектам новую функциональность, оборачивая их в полезные "обёртки". Это гибкая альтернатива наследованию для расширения функциональности.

## Основные концепции Decorator

- **Динамическое добавление ответственности**: Позволяет добавлять новое поведение объектам без изменения их класса
- **Альтернатива наследованию**: В отличие от наследования, декораторы обеспечивают более гибкое расширение функциональности
- **Композиция вместо наследования**: Использует композицию для объединения поведения во время выполнения

## Структура паттерна

- **Компонент (Component)**: Интерфейс для объектов, которые могут иметь добавленные обязанности
- **Конкретный компонент (ConcreteComponent)**: Конкретный объект, который может быть обёрнут декораторами
- **Декоратор (Decorator)**: Хранит ссылку на компонент и реализует его интерфейс
- **Конкретные декораторы (ConcreteDecorators)**: Добавляют конкретную функциональность

## Преимущества

- Большая гибкость, чем у наследования
- Позволяет избежать перегруженных функциями классов на верхних уровнях иерархии
- Можно добавлять и удалять обязанности во время выполнения
- Можно комбинировать несколько декораторов

## Недостатки

- Может привести к большому количеству маленьких классов
- Трудно реализовать декоратор, который не зависит от порядка других декораторов
- Исходный код может быть сложнее для понимания из-за множества обёрток

## Примеры реализации на Python

### 1. Базовый пример с текстом

```python
from abc import ABC, abstractmethod

# Интерфейс компонента
class TextComponent(ABC):
    @abstractmethod
    def render(self) -> str:
        pass

# Конкретный компонент
class PlainText(TextComponent):
    def __init__(self, text):
        self._text = text

    def render(self) -> str:
        return self._text

# Базовый декоратор
class TextDecorator(TextComponent):
    def __init__(self, component: TextComponent):
        self._component = component

    def render(self) -> str:
        return self._component.render()

# Конкретные декораторы
class BoldDecorator(TextDecorator):
    def render(self) -> str:
        return f"<b>{self._component.render()}</b>"

class ItalicDecorator(TextDecorator):
    def render(self) -> str:
        return f"<i>{self._component.render()}</i>"

class UnderlineDecorator(TextDecorator):
    def render(self) -> str:
        return f"<u>{self._component.render()}</u>"

# Использование
text = PlainText("Hello, World!")
decorated_text = BoldDecorator(ItalicDecorator(UnderlineDecorator(text)))

print(decorated_text.render())  # <b><i><u>Hello, World!</u></i></b>
```

### 2. Пример с кофе и добавками

```python
from abc import ABC, abstractmethod

# Абстрактный компонент - напиток
class Beverage(ABC):
    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def cost(self) -> float:
        pass

# Конкретные компоненты - виды кофе
class Espresso(Beverage):
    def get_description(self) -> str:
        return "Espresso"

    def cost(self) -> float:
        return 1.99

class DarkRoast(Beverage):
    def get_description(self) -> str:
        return "Dark Roast Coffee"

    def cost(self) -> float:
        return 0.99

# Абстрактный декоратор - добавка
class CondimentDecorator(Beverage, ABC):
    def __init__(self, beverage: Beverage):
        self._beverage = beverage

    @abstractmethod
    def get_description(self) -> str:
        pass

# Конкретные декораторы - виды добавок
class Milk(CondimentDecorator):
    def get_description(self) -> str:
        return self._beverage.get_description() + ", Milk"

    def cost(self) -> float:
        return self._beverage.cost() + 0.20

class Mocha(CondimentDecorator):
    def get_description(self) -> str:
        return self._beverage.get_description() + ", Mocha"

    def cost(self) -> float:
        return self._beverage.cost() + 0.30

class Whip(CondimentDecorator):
    def get_description(self) -> str:
        return self._beverage.get_description() + ", Whip"

    def cost(self) -> float:
        return self._beverage.cost() + 0.15

# Использование
beverage = Espresso()
print(f"{beverage.get_description()} ${beverage.cost()}")

beverage2 = DarkRoast()
beverage2 = Mocha(beverage2)
beverage2 = Mocha(beverage2)
beverage2 = Whip(beverage2)
print(f"{beverage2.get_description()} ${beverage2.cost()}")
```

### 3. Пример с декораторами функций (встроенная поддержка в Python)

Python имеет встроенную поддержку декораторов для функций:

```python
def make_bold(func):
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper

def make_italic(func):
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper

def make_underline(func):
    def wrapper(*args, **kwargs):
        return f"<u>{func(*args, **kwargs)}</u>"
    return wrapper

# Применение нескольких декораторов
@make_bold
@make_italic
@make_underline
def hello(name):
    return f"Hello, {name}!"

print(hello("World"))  # <b><i><u>Hello, World!</u></i></b>

# Эквивалентно:
# hello = make_bold(make_italic(make_underline(hello)))
```

## Когда использовать Decorator?

- Когда нужно добавлять обязанности объектам динамически и прозрачно
- Когда нельзя расширить функциональность с помощью наследования
- Когда нужно добавлять и удалять обязанности во время выполнения

Decorator особенно полезен в системах, где важно соблюдение принципа открытости/закрытости (классы должны быть открыты для расширения, но закрыты для модификации).
