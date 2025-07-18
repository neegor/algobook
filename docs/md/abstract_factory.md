---
tags: [Патерны проектирования]
---

# Паттерн Abstract Factory (Абстрактная фабрика) в Python

Абстрактная фабрика — это порождающий паттерн проектирования, который позволяет создавать семейства связанных объектов, не привязываясь к конкретным классам создаваемых объектов.

## Основные концепции

1. **Абстрактная фабрика** - интерфейс для создания семейств связанных или зависимых объектов
2. **Конкретная фабрика** - реализует методы абстрактной фабрики, создавая конкретные объекты
3. **Абстрактный продукт** - объявляет интерфейс для типа продукта
4. **Конкретный продукт** - определяет объект продукта, создаваемый соответствующей конкретной фабрикой

## Когда использовать

- Когда система должна быть независимой от процесса создания объектов
- Когда система должна конфигурироваться одним из множества семейств объектов
- Когда семейства связанных объектов должны использоваться вместе
- Когда вы хотите предоставить библиотеку классов, раскрывая только их интерфейсы

## Преимущества

- Изолирует конкретные классы
- Упрощает замену семейств продуктов
- Гарантирует сочетаемость продуктов
- Поддерживает принцип открытости/закрытости

## Недостатки

- Сложность добавления новых видов продуктов
- Может привести к созданию большого числа классов

## Пример 1: Кроссплатформенные UI элементы

```python
from abc import ABC, abstractmethod

# Абстрактные продукты
class Button(ABC):
    @abstractmethod
    def render(self):
        pass

class Checkbox(ABC):
    @abstractmethod
    def render(self):
        pass

# Конкретные продукты для Windows
class WindowsButton(Button):
    def render(self):
        return "Windows Button"

class WindowsCheckbox(Checkbox):
    def render(self):
        return "Windows Checkbox"

# Конкретные продукты для MacOS
class MacOSButton(Button):
    def render(self):
        return "MacOS Button"

class MacOSCheckbox(Checkbox):
    def render(self):
        return "MacOS Checkbox"

# Абстрактная фабрика
class GUIFactory(ABC):
    @abstractmethod
    def create_button(self) -> Button:
        pass

    @abstractmethod
    def create_checkbox(self) -> Checkbox:
        pass

# Конкретные фабрики
class WindowsFactory(GUIFactory):
    def create_button(self) -> Button:
        return WindowsButton()

    def create_checkbox(self) -> Checkbox:
        return WindowsCheckbox()

class MacOSFactory(GUIFactory):
    def create_button(self) -> Button:
        return MacOSButton()

    def create_checkbox(self) -> Checkbox:
        return MacOSCheckbox()

# Клиентский код
def client_code(factory: GUIFactory):
    button = factory.create_button()
    checkbox = factory.create_checkbox()

    print(button.render())
    print(checkbox.render())

# Использование
print("Windows UI:")
client_code(WindowsFactory())

print("\nMacOS UI:")
client_code(MacOSFactory())
```

## Пример 2: Мебель разных стилей

```python
from abc import ABC, abstractmethod

# Абстрактные продукты
class Chair(ABC):
    @abstractmethod
    def sit_on(self):
        pass

class Sofa(ABC):
    @abstractmethod
    def lie_on(self):
        pass

# Конкретные продукты в стиле Викторианский
class VictorianChair(Chair):
    def sit_on(self):
        return "Сидим на викторианском стуле"

class VictorianSofa(Sofa):
    def lie_on(self):
        return "Лежим на викторианском диване"

# Конкретные продукты в стиле Модерн
class ModernChair(Chair):
    def sit_on(self):
        return "Сидим на современном стуле"

class ModernSofa(Sofa):
    def lie_on(self):
        return "Лежим на современном диване"

# Абстрактная фабрика
class FurnitureFactory(ABC):
    @abstractmethod
    def create_chair(self) -> Chair:
        pass

    @abstractmethod
    def create_sofa(self) -> Sofa:
        pass

# Конкретные фабрики
class VictorianFurnitureFactory(FurnitureFactory):
    def create_chair(self) -> Chair:
        return VictorianChair()

    def create_sofa(self) -> Sofa:
        return VictorianSofa()

class ModernFurnitureFactory(FurnitureFactory):
    def create_chair(self) -> Chair:
        return ModernChair()

    def create_sofa(self) -> Sofa:
        return ModernSofa()

# Клиентский код
def furnish_room(factory: FurnitureFactory):
    chair = factory.create_chair()
    sofa = factory.create_sofa()

    print(chair.sit_on())
    print(sofa.lie_on())

# Использование
print("Викторианская комната:")
furnish_room(VictorianFurnitureFactory())

print("\nСовременная комната:")
furnish_room(ModernFurnitureFactory())
```

## Пример 3: Разные типы документов

```python
from abc import ABC, abstractmethod

# Абстрактные продукты
class Document(ABC):
    @abstractmethod
    def open(self):
        pass

class Spreadsheet(ABC):
    @abstractmethod
    def calculate(self):
        pass

# Конкретные продукты для Microsoft Office
class WordDocument(Document):
    def open(self):
        return "Opening Word document"

class ExcelSpreadsheet(Spreadsheet):
    def calculate(self):
        return "Calculating in Excel"

# Конкретные продукты для Google Docs
class GoogleDoc(Document):
    def open(self):
        return "Opening Google Doc"

class GoogleSheet(Spreadsheet):
    def calculate(self):
        return "Calculating in Google Sheets"

# Абстрактная фабрика
class OfficeSuiteFactory(ABC):
    @abstractmethod
    def create_document(self) -> Document:
        pass

    @abstractmethod
    def create_spreadsheet(self) -> Spreadsheet:
        pass

# Конкретные фабрики
class MicrosoftOfficeFactory(OfficeSuiteFactory):
    def create_document(self) -> Document:
        return WordDocument()

    def create_spreadsheet(self) -> Spreadsheet:
        return ExcelSpreadsheet()

class GoogleDocsFactory(OfficeSuiteFactory):
    def create_document(self) -> Document:
        return GoogleDoc()

    def create_spreadsheet(self) -> Spreadsheet:
        return GoogleSheet()

# Клиентский код
def work_with_documents(factory: OfficeSuiteFactory):
    doc = factory.create_document()
    sheet = factory.create_spreadsheet()

    print(doc.open())
    print(sheet.calculate())

# Использование
print("Microsoft Office:")
work_with_documents(MicrosoftOfficeFactory())

print("\nGoogle Docs:")
work_with_documents(GoogleDocsFactory())
```

## Заключение

Паттерн Abstract Factory полезен, когда ваша система должна быть независимой от того, как создаются, компонуются и представляются продукты, или когда вам нужно создавать семейства связанных продуктов. Он инкапсулирует выбор конкретных классов и контролирует их создание.
