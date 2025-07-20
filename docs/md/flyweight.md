---
tags: [Патерны проектирования]
---

# Паттерн Flyweight (Приспособленец)

Паттерн Flyweight - это структурный паттерн проектирования, который позволяет эффективно работать с большим количеством объектов, разделяя общее состояние между ними вместо хранения одинаковых данных в каждом объекте.

## Основная идея

Flyweight используется для минимизации использования памяти или вычислительных затрат путем разделения как можно большего количества данных между подобными объектами. Он особенно полезен, когда:

- В приложении используется большое количество объектов
- Хранение всех этих объектов требует много памяти
- Большая часть состояния объектов может быть вынесена вовне
- После выноса внешнего состояния многие группы объектов могут быть заменены относительно небольшим количеством разделяемых объектов

## Компоненты паттерна

- **Flyweight** - интерфейс, через который легковесные объекты могут получать внешнее состояние
- **ConcreteFlyweight** - реализует интерфейс Flyweight и хранит внутреннее состояние (которое не зависит от контекста)
- **UnsharedConcreteFlyweight** - не все подклассы Flyweight должны быть разделяемыми
- **FlyweightFactory** - создает и управляет flyweight-объектами, обеспечивает правильное разделение flyweight-объектов
- **Client** - хранит внешнее состояние и работает с flyweight-объектами

## Примеры на Python

### Пример 1: Текстовый редактор

Представим текстовый редактор, где каждый символ - это объект. Вместо создания тысячи объектов для каждого символа, мы можем использовать flyweight для разделения общих данных.

```python
import weakref

class CharacterFlyweight:
    _pool = weakref.WeakValueDictionary()

    def __new__(cls, char):
        # Если символ уже есть в пуле, возвращаем его
        obj = cls._pool.get(char)
        if obj is None:
            obj = super().__new__(cls)
            cls._pool[char] = obj
            obj.char = char
        return obj

    def render(self, font, size):
        print(f"Символ '{self.char}' с шрифтом {font} и размером {size}")

class TextEditor:
    def __init__(self):
        self.chars = []

    def add_char(self, char, font, size):
        flyweight = CharacterFlyweight(char)
        self.chars.append((flyweight, font, size))

    def render(self):
        for flyweight, font, size in self.chars:
            flyweight.render(font, size)

# Использование
editor = TextEditor()
editor.add_char('H', 'Arial', 12)
editor.add_char('e', 'Arial', 12)
editor.add_char('l', 'Times New Roman', 14)
editor.add_char('l', 'Times New Roman', 14)
editor.add_char('o', 'Arial', 12)

editor.render()
```

### Пример 2: Игра с деревьями

В игре может быть множество деревьев с одинаковыми текстурами, но разными позициями.

```python
class TreeType:
    def __init__(self, name, color, texture):
        self.name = name
        self.color = color
        self.texture = texture

    def draw(self, x, y):
        print(f"Рисуем дерево {self.name} цвета {self.color} в позиции ({x}, {y})")

class TreeFactory:
    tree_types = {}

    @classmethod
    def get_tree_type(cls, name, color, texture):
        key = (name, color, texture)
        if key not in cls.tree_types:
            cls.tree_types[key] = TreeType(name, color, texture)
        return cls.tree_types[key]

class Tree:
    def __init__(self, x, y, tree_type):
        self.x = x
        self.y = y
        self.type = tree_type

    def draw(self):
        self.type.draw(self.x, self.y)

class Forest:
    def __init__(self):
        self.trees = []

    def plant_tree(self, x, y, name, color, texture):
        tree_type = TreeFactory.get_tree_type(name, color, texture)
        tree = Tree(x, y, tree_type)
        self.trees.append(tree)

    def draw(self):
        for tree in self.trees:
            tree.draw()

# Использование
forest = Forest()
forest.plant_tree(1, 2, "Дуб", "зеленый", "текстура_дуба.png")
forest.plant_tree(3, 4, "Дуб", "зеленый", "текстура_дуба.png")
forest.plant_tree(5, 6, "Береза", "белый", "текстура_березы.png")

forest.draw()
```

### Пример 3: Форматирование текста

```python
class TextStyleFlyweight:
    _pool = {}

    def __new__(cls, font, size, color):
        key = (font, size, color)
        if key not in cls._pool:
            cls._pool[key] = super().__new__(cls)
            cls._pool[key].font = font
            cls._pool[key].size = size
            cls._pool[key].color = color
        return cls._pool[key]

    def apply_style(self, text):
        print(f"Текст: '{text}' | Шрифт: {self.font}, Размер: {self.size}, Цвет: {self.color}")

class FormattedText:
    def __init__(self):
        self.text = []
        self.styles = []

    def add_text(self, text, font=None, size=None, color=None):
        self.text.append(text)
        if font or size or color:
            style = TextStyleFlyweight(font or "Arial", size or 12, color or "black")
            self.styles.append((len(self.text)-1, style))
        else:
            self.styles.append((len(self.text)-1, None))

    def display(self):
        for i, text in enumerate(self.text):
            for pos, style in self.styles:
                if pos == i and style:
                    style.apply_style(text)
                    break
            else:
                print(f"Текст: '{text}' | (стандартное форматирование)")

# Использование
doc = FormattedText()
doc.add_text("Привет, ", "Times New Roman", 14, "red")
doc.add_text("мир!", "Arial", 16, "blue")
doc.add_text(" Это обычный текст.")
doc.add_text(" А это снова стилизованный", "Courier New", 12, "green")

doc.display()
```

## Преимущества и недостатки

**Преимущества:**

- Экономит память за счет разделения общего состояния
- Уменьшает количество создаваемых объектов
- Упрощает работу с большим количеством объектов

**Нестановки:**

- Может увеличить сложность кода
- Требует тщательного разделения внутреннего и внешнего состояния
- Может привести к проблемам с многопоточностью, если flyweight-объекты изменяемы

Flyweight особенно полезен в графических редакторах, играх, текстовых процессорах и других приложениях, где требуется работать с большим количеством похожих объектов.
