---
tags: [Патерны проектирования]
---

# Паттерн Bridge (Мост)

Паттерн Bridge (Мост) - это структурный паттерн проектирования, который разделяет один или несколько классов на две отдельные иерархии - абстракцию и реализацию, позволяя изменять их независимо друг от друга.

## Основная идея

Мост предлагает заменить наследование композицией, поместив реализацию в отдельный класс (или иерархию классов) и передавая экземпляр этого класса в исходный класс в качестве параметра.

## Когда использовать Bridge?

- Когда вы хотите разделить монолитный класс, который содержит несколько различных реализаций какой-то функциональности
- Когда класс нужно расширять в двух независимых плоскостях (абстракция и реализация)
- Когда реализацию нужно изменять во время выполнения программы

## Структура паттерна

```
           Абстракция
          /          \
RefinedAbstractionA   RefinedAbstractionB
         |             |
    Реализация      Реализация
       /    \
ConcreteImplementationA ConcreteImplementationB
```

## Пример 1: Фигуры и способы их рисования

```python
from abc import ABC, abstractmethod

# Реализация (Implementation)
class Renderer(ABC):
    @abstractmethod
    def render_circle(self, radius):
        pass

# Конкретные реализации (Concrete Implementations)
class VectorRenderer(Renderer):
    def render_circle(self, radius):
        print(f"Drawing a circle of radius {radius} using vector graphics")

class RasterRenderer(Renderer):
    def render_circle(self, radius):
        print(f"Drawing a circle of radius {radius} using pixels")

# Абстракция (Abstraction)
class Shape:
    def __init__(self, renderer):
        self.renderer = renderer

    def draw(self): pass
    def resize(self, factor): pass

# Уточненная абстракция (Refined Abstraction)
class Circle(Shape):
    def __init__(self, renderer, radius):
        super().__init__(renderer)
        self.radius = radius

    def draw(self):
        self.renderer.render_circle(self.radius)

    def resize(self, factor):
        self.radius *= factor

# Клиентский код
if __name__ == "__main__":
    raster = RasterRenderer()
    vector = VectorRenderer()

    circle1 = Circle(raster, 5)
    circle1.draw()
    circle1.resize(2)
    circle1.draw()

    circle2 = Circle(vector, 10)
    circle2.draw()
```

Вывод:

```
Drawing a circle of radius 5 using pixels
Drawing a circle of radius 10 using pixels
Drawing a circle of radius 10 using vector graphics
```

## Пример 2: Устройства и пульты управления

```python
from abc import ABC, abstractmethod

# Реализация (устройства)
class Device(ABC):
    @property
    @abstractmethod
    def is_enabled(self):
        pass

    @abstractmethod
    def enable(self):
        pass

    @abstractmethod
    def disable(self):
        pass

    @abstractmethod
    def get_volume(self):
        pass

    @abstractmethod
    def set_volume(self, percent):
        pass

    @abstractmethod
    def get_channel(self):
        pass

    @abstractmethod
    def set_channel(self, channel):
        pass

# Конкретные устройства
class TV(Device):
    def __init__(self):
        self._on = False
        self._volume = 50
        self._channel = 1

    @property
    def is_enabled(self):
        return self._on

    def enable(self):
        self._on = True

    def disable(self):
        self._on = False

    def get_volume(self):
        return self._volume

    def set_volume(self, percent):
        self._volume = percent

    def get_channel(self):
        return self._channel

    def set_channel(self, channel):
        self._channel = channel

class Radio(Device):
    def __init__(self):
        self._on = False
        self._volume = 30
        self._channel = 101.5

    @property
    def is_enabled(self):
        return self._on

    def enable(self):
        self._on = True

    def disable(self):
        self._on = False

    def get_volume(self):
        return self._volume

    def set_volume(self, percent):
        self._volume = percent

    def get_channel(self):
        return self._channel

    def set_channel(self, channel):
        self._channel = channel

# Абстракция (пульты)
class Remote:
    def __init__(self, device: Device):
        self._device = device

    def toggle_power(self):
        if self._device.is_enabled:
            self._device.disable()
        else:
            self._device.enable()

    def volume_down(self):
        self._device.set_volume(self._device.get_volume() - 10)

    def volume_up(self):
        self._device.set_volume(self._device.get_volume() + 10)

    def channel_down(self):
        self._device.set_channel(self._device.get_channel() - 1)

    def channel_up(self):
        self._device.set_channel(self._device.get_channel() + 1)

# Уточненная абстракция (расширенный пульт)
class AdvancedRemote(Remote):
    def mute(self):
        self._device.set_volume(0)

# Клиентский код
if __name__ == "__main__":
    tv = TV()
    remote = Remote(tv)
    remote.toggle_power()
    remote.volume_up()
    remote.channel_up()
    print(f"TV: channel={tv.get_channel()}, volume={tv.get_volume()}")

    radio = Radio()
    advanced_remote = AdvancedRemote(radio)
    advanced_remote.toggle_power()
    advanced_remote.mute()
    print(f"Radio: channel={radio.get_channel()}, volume={radio.get_volume()}")
```

Вывод:

```
TV: channel=2, volume=60
Radio: channel=101.5, volume=0
```

## Преимущества паттерна Bridge

- Разделяет абстракцию и реализацию, позволяя изменять их независимо
- Уменьшает количество подклассов (нет необходимости создавать комбинации абстракций и реализаций)
- Позволяет добавлять новые абстракции и реализации независимо
- Позволяет скрыть детали реализации от клиентского кода

## Отличие от других паттернов

- **Adapter** пытается сделать интерфейсы совместимыми, а **Bridge** разделяет абстракцию и реализацию заранее
- **Abstract Factory** может работать вместе с Bridge для создания конкретных реализаций
- **Strategy** похож на Bridge, но фокусируется на изменении поведения, а Bridge - на структуре

Bridge особенно полезен в ситуациях, когда у вас есть несколько вариантов абстракции и несколько вариантов реализации, и вы хотите избежать экспоненциального роста количества классов при их комбинировании.
