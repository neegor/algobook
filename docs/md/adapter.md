---
tags: [Патерны проектирования]
---

# Adapter (Адаптер)

**Adapter** (Адаптер) - это структурный паттерн проектирования, который позволяет объектам с несовместимыми интерфейсами работать вместе. Он действует как мост между двумя несовместимыми интерфейсами, преобразуя интерфейс одного класса в интерфейс, ожидаемый клиентом.

Адаптер часто используется, когда:

- Нужно использовать существующий класс, но его интерфейс не соответствует требуемому
- Нужно создать повторно используемый класс, который сотрудничает с классами, имеющими несовместимые интерфейсы
- Нужно предоставить несколько интерфейсов для одного класса

## Типы адаптеров

1. **Адаптер класса** (использует множественное наследование)
2. **Адаптер объекта** (использует композицию)

## Примеры на Python

### 1. Пример с адаптером объекта

```python
# Целевой интерфейс, который ожидает клиент
class Target:
    def request(self) -> str:
        return "Целевое поведение"

# Класс с несовместимым интерфейсом
class Adaptee:
    def specific_request(self) -> str:
        return ".еинедепов еономис яамаД"

# Адаптер, преобразующий интерфейс Adaptee в Target
class Adapter(Target):
    def __init__(self, adaptee: Adaptee) -> None:
        self.adaptee = adaptee

    def request(self) -> str:
        return f"Адаптер: (ПЕРЕВЕДЕНО) {self.adaptee.specific_request()[::-1]}"

# Клиентский код
def client_code(target: Target) -> None:
    print(target.request(), end="")

if __name__ == "__main__":
    print("Клиент: Я работаю с объектами Target:")
    target = Target()
    client_code(target)
    print("\n")

    adaptee = Adaptee()
    print("Клиент: У Adaptee странный интерфейс. Я его не понимаю:")
    print(f"Adaptee: {adaptee.specific_request()}", end="\n\n")

    print("Клиент: Но я могу работать с ним через Adapter:")
    adapter = Adapter(adaptee)
    client_code(adapter)
```

Вывод:

```
Клиент: Я работаю с объектами Target:
Целевое поведение

Клиент: У Adaptee странный интерфейс. Я его не понимаю:
Adaptee: .еинедепов еономис яамаД

Клиент: Но я могу работать с ним через Adapter:
Адаптер: (ПЕРЕВЕДЕНО) Дамия симое поведение.
```

### 2. Практический пример с платежными системами

```python
# Целевой интерфейс для обработки платежей
class PaymentProcessor:
    def pay(self, amount: float) -> None:
        pass

# Старая система с несовместимым интерфейсом
class LegacyPaymentSystem:
    def make_payment(self, dollars: int, cents: int) -> None:
        print(f"Оплата через Legacy систему: {dollars} долларов и {cents} центов")

# Адаптер для старой системы
class LegacyPaymentAdapter(PaymentProcessor):
    def __init__(self, legacy_system: LegacyPaymentSystem):
        self.legacy_system = legacy_system

    def pay(self, amount: float) -> None:
        dollars = int(amount)
        cents = int((amount - dollars) * 100)
        self.legacy_system.make_payment(dollars, cents)

# Новая система, совместимая с целевым интерфейсом
class ModernPaymentSystem(PaymentProcessor):
    def pay(self, amount: float) -> None:
        print(f"Оплата через Modern систему: {amount:.2f} USD")

# Клиентский код
def process_payment(processor: PaymentProcessor, amount: float):
    processor.pay(amount)

if __name__ == "__main__":
    modern_system = ModernPaymentSystem()
    legacy_system = LegacyPaymentSystem()
    legacy_adapter = LegacyPaymentAdapter(legacy_system)

    print("Используем Modern систему:")
    process_payment(modern_system, 123.45)

    print("\nИспользуем Legacy систему через адаптер:")
    process_payment(legacy_adapter, 123.45)
```

Вывод:

```
Используем Modern систему:
Оплата через Modern систему: 123.45 USD

Используем Legacy систему через адаптер:
Оплата через Legacy систему: 123 долларов и 45 центов
```

### 3. Пример с адаптером класса (множественное наследование)

```python
# Целевой интерфейс
class MediaPlayer:
    def play(self, audio_type: str, file_name: str) -> None:
        pass

# Адаптируемый класс 1
class AdvancedMediaPlayer:
    def play_vlc(self, file_name: str) -> None:
        print(f"Воспроизведение VLC файла: {file_name}")

    def play_mp4(self, file_name: str) -> None:
        print(f"Воспроизведение MP4 файла: {file_name}")

# Адаптер класса (использует множественное наследование)
class MediaAdapter(MediaPlayer, AdvancedMediaPlayer):
    def play(self, audio_type: str, file_name: str) -> None:
        if audio_type == "vlc":
            self.play_vlc(file_name)
        elif audio_type == "mp4":
            self.play_mp4(file_name)
        else:
            raise ValueError(f"Не поддерживаемый формат: {audio_type}")

# Конкретная реализация MediaPlayer
class AudioPlayer(MediaPlayer):
    def play(self, audio_type: str, file_name: str) -> None:
        if audio_type == "mp3":
            print(f"Воспроизведение MP3 файла: {file_name}")
        elif audio_type in ["vlc", "mp4"]:
            adapter = MediaAdapter()
            adapter.play(audio_type, file_name)
        else:
            raise ValueError(f"Не поддерживаемый формат: {audio_type}")

# Клиентский код
if __name__ == "__main__":
    player = AudioPlayer()

    player.play("mp3", "song.mp3")
    player.play("mp4", "movie.mp4")
    player.play("vlc", "series.vlc")
```

Вывод:

```
Воспроизведение MP3 файла: song.mp3
Воспроизведение MP4 файла: movie.mp4
Воспроизведение VLC файла: series.vlc
```

## Преимущества и недостатки

**Преимущества:**

- Позволяет повторно использовать существующие классы
- Разделяет и скрывает от клиента подробности преобразования интерфейсов
- Реализует принцип открытости/закрытости (можно вводить новые адаптеры без изменения клиентского кода)

**Недостатки:**

- Усложняет код из-за введения дополнительных классов
- В случае адаптера класса невозможно адаптировать класс и его подклассы одновременно

## Когда использовать Adapter?

- Когда нужно использовать существующий класс, но его интерфейс не соответствует вашим потребностям
- Когда нужно создать класс, который должен взаимодействовать с классами, имеющими несовместимые интерфейсы
- Когда нужно обеспечить работу нескольких подклассов, но impractical адаптировать их интерфейсы путем переопределения всех методов

Adapter особенно полезен при интеграции legacy-кода или при работе со сторонними библиотеками, интерфейсы которых не соответствуют вашим требованиям.
