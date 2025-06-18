# Merkle Patricia Trie (MPT)

## Введение

**Merkle Patricia Trie (MPT)** - это гибридная структура данных, сочетающая преимущества **Patricia Trie** (префиксного дерева) и хеш-дерева Меркла. Это фундаментальная структура данных в блокчейн-системах, особенно в Ethereum.

**Merkle Patricia Trie** - это мощная структура данных, сочетающая преимущества префиксных деревьев для эффективного поиска и хеш-деревьев для проверки целостности. Ее применение в блокчейне Ethereum демонстрирует, как можно эффективно хранить и проверять большие объемы данных в распределенных системах.

Приведенная реализация является упрощенной. В production-системах (как Ethereum) используются дополнительные оптимизации и особенности реализации.

## Основные компоненты

### 1. Patricia Trie (Префиксное дерево)

Это сжатое префиксное дерево, где узлы с единственным потомком объединяются с родительским узлом. Это экономит пространство и ускоряет поиск.

### 2. Хеш-дерево Меркла

Каждый узел содержит криптографический хеш своих данных и хешей дочерних узлов, что обеспечивает целостность данных.

## Типы узлов в MPT

1. **Пустой узел (Empty)**: представляет пустую строку.
2. **Лист (Leaf)**: содержит ключ и значение, представляет конечную точку в дереве.
3. **Расширение (Extension)**: содержит общий префикс и ссылку на следующий узел.
4. **Ветвь (Branch)**: узел с 16 дочерними элементами (для 16-ричной системы) и возможным значением.

## Применение

1. **Блокчейн Ethereum**:

   - Хранение состояния (аккаунты, балансы)
   - Хранение транзакций и квитанций
   - Быстрая верификация данных без полной загрузки состояния

2. **Распределенные системы**:

   - Проверка целостности данных
   - Эффективные доказательства включения/исключения

3. **Системы контроля версий**:
   - Эффективное хранение и проверка истории изменений

## Реализация на Python

Вот упрощенная реализация **Merkle Patricia Trie** на Python:

```python
import rlp
from hashlib import sha3_256
from typing import Dict, List, Optional, Tuple

# Определение типов узлов
Nibbles = List[int]
Node = List[bytes]

# Кодирование/декодирование RLP
def encode_node(node: Node) -> bytes:
    return rlp.encode(node)

def decode_node(data: bytes) -> Node:
    return rlp.decode(data)

# Хеширование
def keccak256(data: bytes) -> bytes:
    return sha3_256(data).digest()

class MerklePatriciaTrie:
    def __init__(self, storage: Dict[bytes, bytes], root_hash: Optional[bytes] = None):
        self._storage = storage
        self.root_hash = root_hash

    def get(self, key: bytes) -> Optional[bytes]:
        return self._get(self.root_hash, self.bytes_to_nibbles(key))

    def _get(self, node_hash: Optional[bytes], nibbles: Nibbles) -> Optional[bytes]:
        if node_hash is None:
            return None

        node = decode_node(self._storage[node_hash])

        if len(node) == 2:  # Лист или расширение
            path, rest = self.decode_path(node[0])
            if nibbles[:len(path)] == path:
                if self.is_leaf(node[0]):
                    if len(nibbles) == len(path):
                        return node[1]
                else:  # Расширение
                    return self._get(node[1], nibbles[len(path):])
        elif len(node) == 17:  # Ветвь
            if not nibbles:
                return node[16] if node[16] else None
            child_hash = node[nibbles[0]]
            return self._get(child_hash, nibbles[1:])

        return None

    def put(self, key: bytes, value: bytes):
        nibbles = self.bytes_to_nibbles(key)
        self.root_hash, _ = self._put(self.root_hash, nibbles, value)

    def _put(self, node_hash: Optional[bytes], nibbles: Nibbles, value: bytes) -> Tuple[bytes, bool]:
        if node_hash is None:
            # Создаем новый лист
            new_node = [self.encode_path(nibbles, True), value]
            new_node_encoded = encode_node(new_node)
            new_node_hash = keccak256(new_node_encoded)
            self._storage[new_node_hash] = new_node_encoded
            return new_node_hash, True

        node = decode_node(self._storage[node_hash])

        if len(node) == 2:  # Лист или расширение
            path, _ = self.decode_path(node[0])
            common_prefix = self.find_common_prefix(nibbles, path)

            if common_prefix == len(path):
                if self.is_leaf(node[0]):
                    # Обновляем существующий лист
                    new_node = [node[0], value]
                    new_node_encoded = encode_node(new_node)
                    new_node_hash = keccak256(new_node_encoded)
                    self._storage[new_node_hash] = new_node_encoded
                    return new_node_hash, True
                else:  # Расширение
                    child_hash, updated = self._put(node[1], nibbles[len(path):], value)
                    if updated:
                        new_node = [node[0], child_hash]
                        new_node_encoded = encode_node(new_node)
                        new_node_hash = keccak256(new_node_encoded)
                        self._storage[new_node_hash] = new_node_encoded
                        return new_node_hash, True
                    return node_hash, False
            else:
                # Нужно разделить узел
                branch_node = [b''] * 17
                if len(path) == common_prefix + 1:
                    branch_node[path[common_prefix]] = node[1]
                else:
                    extension_path = path[common_prefix+1:]
                    extension_node = [
                        self.encode_path(extension_path, self.is_leaf(node[0])),
                        node[1]
                    ]
                    extension_encoded = encode_node(extension_node)
                    extension_hash = keccak256(extension_encoded)
                    self._storage[extension_hash] = extension_encoded
                    branch_node[path[common_prefix]] = extension_hash

                if len(nibbles) == common_prefix:
                    branch_node[16] = value
                else:
                    leaf_path = nibbles[common_prefix+1:]
                    leaf_node = [
                        self.encode_path(leaf_path, True),
                        value
                    ]
                    leaf_encoded = encode_node(leaf_node)
                    leaf_hash = keccak256(leaf_encoded)
                    self._storage[leaf_hash] = leaf_encoded
                    branch_node[nibbles[common_prefix]] = leaf_hash

                if common_prefix == 0:
                    branch_encoded = encode_node(branch_node)
                    branch_hash = keccak256(branch_encoded)
                    self._storage[branch_hash] = branch_encoded
                    return branch_hash, True
                else:
                    extension_node = [
                        self.encode_path(path[:common_prefix], False),
                        branch_hash
                    ]
                    extension_encoded = encode_node(extension_node)
                    extension_hash = keccak256(extension_encoded)
                    self._storage[extension_hash] = extension_encoded
                    return extension_hash, True
        elif len(node) == 17:  # Ветвь
            if not nibbles:
                if node[16] == value:
                    return node_hash, False
                new_node = node.copy()
                new_node[16] = value
                new_node_encoded = encode_node(new_node)
                new_node_hash = keccak256(new_node_encoded)
                self._storage[new_node_hash] = new_node_encoded
                return new_node_hash, True
            else:
                child_hash, updated = self._put(node[nibbles[0]], nibbles[1:], value)
                if updated:
                    new_node = node.copy()
                    new_node[nibbles[0]] = child_hash
                    new_node_encoded = encode_node(new_node)
                    new_node_hash = keccak256(new_node_encoded)
                    self._storage[new_node_hash] = new_node_encoded
                    return new_node_hash, True
                return node_hash, False

        raise Exception("Invalid node")

    @staticmethod
    def bytes_to_nibbles(key: bytes) -> Nibbles:
        nibbles = []
        for byte in key:
            nibbles.append(byte >> 4)
            nibbles.append(byte & 0x0F)
        return nibbles

    @staticmethod
    def nibbles_to_bytes(nibbles: Nibbles) -> bytes:
        if len(nibbles) % 2 != 0:
            raise ValueError("Nibbles must be even length")
        bytes_list = []
        for i in range(0, len(nibbles), 2):
            byte = (nibbles[i] << 4) | nibbles[i+1]
            bytes_list.append(byte)
        return bytes(bytes_list)

    @staticmethod
    def encode_path(path: Nibbles, is_leaf: bool) -> bytes:
        flags = 0x20 if is_leaf else 0x00
        if len(path) % 2 != 0:
            flags |= 0x10
            first_nibble = path[0]
            path = path[1:]
        else:
            first_nibble = 0

        first_byte = flags | first_nibble
        path_bytes = [first_byte] + path
        return bytes(path_bytes)

    @staticmethod
    def decode_path(path: bytes) -> Tuple[Nibbles, bool]:
        if not path:
            return [], False

        first_byte = path[0]
        is_leaf = (first_byte & 0x20) != 0
        has_odd_nibble = (first_byte & 0x10) != 0

        nibbles = []
        if has_odd_nibble:
            nibbles.append(first_byte & 0x0F)

        for byte in path[1:]:
            nibbles.append(byte >> 4)
            nibbles.append(byte & 0x0F)

        return nibbles, is_leaf

    @staticmethod
    def is_leaf(path: bytes) -> bool:
        return (path[0] & 0x20) != 0

    @staticmethod
    def find_common_prefix(a: Nibbles, b: Nibbles) -> int:
        length = min(len(a), len(b))
        for i in range(length):
            if a[i] != b[i]:
                return i
        return length
```

## Пример использования

```python
# Инициализация хранилища и дерева
storage = {}
trie = MerklePatriciaTrie(storage)

# Добавление данных
trie.put(b'key1', b'value1')
trie.put(b'key2', b'value2')
trie.put(b'key3', b'value3')

# Получение данных
print(trie.get(b'key1'))  # b'value1'
print(trie.get(b'key2'))  # b'value2'
print(trie.get(b'key4'))  # None

# Корневой хеш изменяется при изменении данных
old_root = trie.root_hash
trie.put(b'key1', b'new_value1')
print(trie.root_hash != old_root)  # True
```

## Оптимизации и особенности

1. **16-ричное представление**: Ключи преобразуются в последовательность "нибблов" (4-битных значений) для эффективного хранения.
2. **Кодирование пути**: Пути в узлах кодируются с флагами, указывающими тип узла (лист или расширение) и четность длины.
3. **Сжатие путей**: Последовательные узлы с одним потомком объединяются в расширения.
4. **Хеширование**: Только измененные узлы пересчитываются при обновлении, что обеспечивает эффективность.
