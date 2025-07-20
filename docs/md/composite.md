---
tags: [Патерны проектирования]
---

# Паттерн Composite (Компоновщик)

Composite - это структурный паттерн проектирования, который позволяет сгруппировать множество объектов в древовидную структуру, а затем работать с ней так, как будто это единый объект.

Основная идея: клиенты могут единообразно обрабатывать как отдельные объекты, так и их композиции.

## Когда использовать

- Когда вам нужно представить древовидную структуру объектов
- Когда вы хотите, чтобы клиенты могли единообразно работать как с простыми, так и со сложными компонентами

## Структура

```
Component (абстрактный)
│
├── Leaf (лист)
└── Composite (композит)
    ├── children: Component[]
    └── operation()
```

## Примеры реализации на Python

### Пример 1: Графические примитивы

```python
from abc import ABC, abstractmethod
from typing import List

# Абстрактный компонент
class Graphic(ABC):
    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def add(self, graphic):
        pass

    @abstractmethod
    def remove(self, graphic):
        pass

    @abstractmethod
    def get_child(self, index):
        pass

# Лист (простой компонент)
class Circle(Graphic):
    def draw(self):
        print("Рисуем круг")

    def add(self, graphic):
        raise NotImplementedError("Нельзя добавить к кругу")

    def remove(self, graphic):
        raise NotImplementedError("Нельзя удалить из круга")

    def get_child(self, index):
        raise NotImplementedError("У круга нет потомков")

# Лист (простой компонент)
class Square(Graphic):
    def draw(self):
        print("Рисуем квадрат")

    def add(self, graphic):
        raise NotImplementedError("Нельзя добавить к квадрату")

    def remove(self, graphic):
        raise NotImplementedError("Нельзя удалить из квадрата")

    def get_child(self, index):
        raise NotImplementedError("У квадрата нет потомков")

# Композит (составной компонент)
class CompositeGraphic(Graphic):
    def __init__(self):
        self._children: List[Graphic] = []

    def draw(self):
        print("Рисуем композитную графику:")
        for child in self._children:
            child.draw()

    def add(self, graphic):
        self._children.append(graphic)

    def remove(self, graphic):
        self._children.remove(graphic)

    def get_child(self, index):
        return self._children[index]

# Использование
circle1 = Circle()
circle2 = Circle()
square = Square()

composite1 = CompositeGraphic()
composite1.add(circle1)
composite1.add(circle2)

composite2 = CompositeGraphic()
composite2.add(square)
composite2.add(composite1)

composite2.draw()
```

### Пример 2: Файловая система

```python
from abc import ABC, abstractmethod
from typing import List

# Компонент
class FileSystemComponent(ABC):
    @abstractmethod
    def show_info(self):
        pass

# Лист (файл)
class File(FileSystemComponent):
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def show_info(self):
        print(f"Файл: {self.name}, Размер: {self.size} KB")

# Композит (папка)
class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children: List[FileSystemComponent] = []

    def add(self, component):
        self.children.append(component)

    def remove(self, component):
        self.children.remove(component)

    def show_info(self):
        print(f"Папка: {self.name}")
        print("Содержимое:")
        for child in self.children:
            child.show_info()

# Использование
file1 = File("document.txt", 100)
file2 = File("image.jpg", 500)
file3 = File("data.csv", 50)

dir1 = Directory("Documents")
dir1.add(file1)
dir1.add(file2)

dir2 = Directory("Project")
dir2.add(file3)
dir2.add(dir1)

dir2.show_info()
```

### Пример 3: Меню ресторана

```python
from abc import ABC, abstractmethod
from typing import List

# Компонент
class MenuComponent(ABC):
    @abstractmethod
    def print(self):
        pass

# Лист (пункт меню)
class MenuItem(MenuComponent):
    def __init__(self, name, price):
        self.name = name
        self.price = price

    def print(self):
        print(f"  {self.name} - ${self.price}")

# Композит (меню)
class Menu(MenuComponent):
    def __init__(self, name):
        self.name = name
        self.children: List[MenuComponent] = []

    def add(self, component):
        self.children.append(component)

    def print(self):
        print(f"\n{self.name}")
        print("----------------")
        for child in self.children:
            child.print()

# Использование
breakfast_menu = Menu("Завтрак")
breakfast_menu.add(MenuItem("Омлет", 5.99))
breakfast_menu.add(MenuItem("Блинчики", 4.50))

dinner_menu = Menu("Обед")
dinner_menu.add(MenuItem("Стейк", 12.99))
dinner_menu.add(MenuItem("Салат", 7.50))

dessert_menu = Menu("Десерты")
dessert_menu.add(MenuItem("Чизкейк", 4.99))
dessert_menu.add(MenuItem("Мороженое", 3.50))

main_menu = Menu("Главное меню")
main_menu.add(breakfast_menu)
main_menu.add(dinner_menu)
main_menu.add(dessert_menu)

main_menu.print()
```

## Преимущества и недостатки

**Преимущества:**

- Упрощает архитектуру клиентского кода
- Облегчает добавление новых типов компонентов
- Позволяет работать с древовидными структурами

**Недостатки:**

- Может сделать дизайн слишком общим (иногда сложно ограничить компоненты)
- Может быть сложно обеспечить соблюдение ограничений для листьев

## Заключение

Паттерн Composite особенно полезен, когда вам нужно работать с иерархическими структурами, где одни и те же операции могут быть применены как к отдельным объектам, так и к их группам. В Python его реализация довольно проста благодаря динамической типизации и поддержке абстрактных базовых классов.
