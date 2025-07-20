---
tags: [Патерны проектирования]
---

# Паттерн Facade (Фасад)

Facade (Фасад) - это структурный паттерн проектирования, который предоставляет простой интерфейс к сложной системе классов, библиотеке или фреймворку. Фасад скрывает сложность внутренней системы и предоставляет клиенту только необходимый функционал.

## Когда использовать Facade?

- Когда вам нужно представить простой или урезанный интерфейс к сложной подсистеме
- Когда вы хотите разложить подсистему на отдельные слои (слои абстракции)
- Когда нужно уменьшить количество зависимостей между клиентом и сложной системой

## Преимущества Facade

- Изолирует клиентов от компонентов сложной подсистемы
- Уменьшает coupling (связность) между клиентским кодом и подсистемой
- Делает подсистему проще в использовании и понимании

## Недостатки Facade

- Фасад может стать "божественным объектом", привязанным ко всем классам программы

## Пример 1: Упрощение работы с мультимедийной системой

```python
# Сложные подсистемы
class VideoFile:
    def __init__(self, filename):
        self.filename = filename

class CodecFactory:
    @staticmethod
    def extract(file):
        print(f"Extracting codec from {file.filename}")
        return "codec"

class BitrateReader:
    @staticmethod
    def read(file, codec):
        print(f"Reading file {file.filename} with {codec}")
        return "buffer"

class AudioMixer:
    @staticmethod
    def fix(buffer):
        print("Fixing audio")
        return "fixed_audio"

# Фасад
class VideoConverter:
    def convert(self, filename, format):
        print("VideoConversionFacade: conversion started.")
        file = VideoFile(filename)
        codec = CodecFactory.extract(file)
        buffer = BitrateReader.read(file, codec)

        if format == "mp4":
            print("Converting to MP4 format")
        else:
            print("Converting to OGG format")

        result = AudioMixer.fix(buffer)
        print("VideoConversionFacade: conversion completed.")
        return result

# Клиентский код
if __name__ == "__main__":
    converter = VideoConverter()
    mp4 = converter.convert("youtubevideo.ogg", "mp4")
```

## Пример 2: Упрощение работы с компьютером

```python
# Сложные подсистемы
class CPU:
    def execute(self):
        print("CPU: Executing instructions")

    def halt(self):
        print("CPU: Halting")

class Memory:
    def load(self, position, data):
        print(f"Memory: Loading data '{data}' at position {position}")

class HardDrive:
    def read(self, lba, size):
        print(f"HardDrive: Reading sector {lba} with size {size}")
        return "boot_data"

# Фасад
class Computer:
    def __init__(self):
        self.cpu = CPU()
        self.memory = Memory()
        self.hard_drive = HardDrive()

    def start(self):
        print("Computer: Starting...")
        boot_data = self.hard_drive.read(0, 1024)
        self.memory.load(0, boot_data)
        self.cpu.execute()
        print("Computer: Started successfully")

    def shutdown(self):
        print("Computer: Shutting down...")
        self.cpu.halt()
        print("Computer: Shutdown complete")

# Клиентский код
if __name__ == "__main__":
    computer = Computer()
    computer.start()
    print("\nUsing computer...\n")
    computer.shutdown()
```

## Пример 3: Упрощение работы с банковской системой

```python
# Сложные подсистемы
class AccountManager:
    def check_account(self, account_id):
        print(f"Checking account {account_id} exists")
        return True

class BalanceChecker:
    def get_balance(self, account_id):
        print(f"Getting balance for account {account_id}")
        return 1000.0

class TransactionProcessor:
    def deposit(self, account_id, amount):
        print(f"Depositing {amount} to account {account_id}")

    def withdraw(self, account_id, amount):
        print(f"Withdrawing {amount} from account {account_id}")

class SecurityManager:
    def verify_pin(self, account_id, pin):
        print(f"Verifying PIN for account {account_id}")
        return pin == "1234"

# Фасад
class BankFacade:
    def __init__(self):
        self.account_manager = AccountManager()
        self.balance_checker = BalanceChecker()
        self.transaction_processor = TransactionProcessor()
        self.security_manager = SecurityManager()

    def deposit_money(self, account_id, pin, amount):
        if not self.security_manager.verify_pin(account_id, pin):
            print("Invalid PIN")
            return False

        if not self.account_manager.check_account(account_id):
            print("Account not found")
            return False

        self.transaction_processor.deposit(account_id, amount)
        print(f"Successfully deposited {amount} to account {account_id}")
        return True

    def withdraw_money(self, account_id, pin, amount):
        if not self.security_manager.verify_pin(account_id, pin):
            print("Invalid PIN")
            return False

        if not self.account_manager.check_account(account_id):
            print("Account not found")
            return False

        balance = self.balance_checker.get_balance(account_id)
        if balance < amount:
            print("Insufficient funds")
            return False

        self.transaction_processor.withdraw(account_id, amount)
        print(f"Successfully withdrew {amount} from account {account_id}")
        return True

# Клиентский код
if __name__ == "__main__":
    bank = BankFacade()

    # Успешное снятие денег
    bank.withdraw_money("acc123", "1234", 500)

    print("\n")

    # Неудачная попытка (неправильный PIN)
    bank.deposit_money("acc123", "1111", 200)
```

## Заключение

Паттерн Facade полезен, когда вам нужно:

- Предоставить простой интерфейс к сложной системе
- Уменьшить зависимости между клиентским кодом и подсистемой
- Организовать подсистему в слои

Фасад не запрещает прямой доступ к классам подсистемы, если это необходимо, но предоставляет удобный способ работы для большинства клиентов.
