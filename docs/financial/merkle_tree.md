# Дерево Меркла (дерево хешей)

В области криптографии и информатики хэш-дерево, также известное как дерево `Меркла`, представляет собой древовидную структуру, в которой каждый узел помечен криптографическим хэш-кодом блока данных. Каждый узел, будь то ветвь, внутренний узел или индексный узел, содержит криптографический хэш-код блока данных и метки своих дочерних узлов.

Хэш-дерево позволяет эффективно и безопасно проверять содержимое больших структур данных. Оно является обобщением хэш-списка и хэш-цепочки.
Чтобы продемонстрировать, что конечный узел является частью данного двоичного хэш-дерева, необходимо вычислить количество хэшей, пропорциональное логарифму числа конечных узлов в дереве. В отличие от хэш-списка, где это число пропорционально количеству конечных узлов, в дереве `Меркла` сложность вычислений уменьшается.

Таким образом, дерево `Меркла` служит эффективным примером криптографической схемы принятия обязательств. Корень дерева рассматривается как обязательство, а конечные узлы могут быть обнаружены и доказано, что они являются частью первоначального обязательства.

Концепция хэш-дерева была изобретена Ральфом Мерклом в 1979 году и запатентована.

## Описание

Заголовок блока в системе Биткойн содержит значение `merkle_root` — хэш всех транзакций в этом блоке. Это предоставляет несколько важных преимуществ и помогает снизить общую нагрузку на сеть.

После того как накапливается достаточное количество блоков, старые транзакции могут быть удалены для экономии места. При этом заголовок блока остаётся неизменным, поскольку содержит только `merkle_root`. Блок без транзакций занимает всего 80 байт, что составляет примерно 4,2 мегабайта в год, если блок генерируется каждые 10 минут.

Благодаря этому становится возможной упрощённая проверка оплаты (англ. `simplified payment verification`, `SPV`). SPV-узел загружает не весь блок, а только его заголовок. Для интересующей его транзакции он запрашивает также её аутентификационный путь. Таким образом, он загружает всего несколько килобайт, тогда как полный размер блока может превышать 10 мегабайт.

Использование этого метода требует, чтобы пользователь доверял узлу сети, у которого будет запрашивать заголовки блоков. Один из способов избежать атаки, то есть подмены узла недобросовестной стороной, — рассылать оповещения по всей сети при обнаружении ошибки в блоке, вынуждая пользователя загружать блок целиком.

На упрощённой проверке оплаты основаны так называемые «тонкие» биткойн-клиенты.

`Хеш-деревья` имеют преимущество перед `хеш-цепочками` или `хеш-функциями`. При использовании `хеш-деревьев` гораздо менее затратным является доказательство принадлежности определённого блока данных к множеству. Поскольку различными блоками часто являются независимые данные, такие как транзакции или части файлов, то нас интересует возможность проверить только один блок, не пересчитывая хеши для остальных узлов дерева.

В общем случае можно записать:

$$ \operatorname{signature}(L) = (L \mid \mathrm{auth}_{L_1}, \dots, \mathrm{auth}_{L_{K-1}}), $$

а проверку осуществить как $\mathrm{TopHash} = \mathrm{Hash}_K$, где

$$ \mathrm{Hash}_k = \begin{cases}
  \operatorname{hash}(L),  & \text{если } k = 1, \\
  \operatorname{hash}(\mathrm{Hash}_{k-1} + \mathrm{auth}_{L_{k-1}}), & \text{если } 2 \leqslant k \leqslant K.
\end{cases} $$

Набор блоков $\{\mathrm{auth}_{L_1}, \dots, \mathrm{auth}_{L_{K-1}}\}$ называется аутентификационный путь или путь Меркла.


## Реализация

```python title="python"

from typing import List
import typing
import hashlib


class Node:
    # (1)!
    def __init__(self, left, right, value: str) -> None:
        self.left: Node = left
        self.right: Node = right
        self.value = value

    @staticmethod
    def hash(val: str) -> str:
        return hashlib.sha256(val.encode("utf-8")).hexdigest()

    @staticmethod
    def doubleHash(val: str) -> str:
        return Node.hash(Node.hash(val))


class MerkleTree:
    # (2)!
    def __init__(self, values: List[str]) -> None:
        self.__buildTree(values)

    def __buildTree(self, values: List[str]) -> None:
        leaves: List[Node] = [Node(None, None, Node.doubleHash(e)) for e in values]
        self.root: Node = self.__buildTreeRec(leaves)

    def __buildTreeRec(self, nodes: List[Node]) -> Node:
        half: int = len(nodes) // 2

        if len(nodes) == 1:
            return Node(nodes[0], nodes[0], Node.doubleHash(nodes[0].value))

        if len(nodes) == 2:
            return Node(
                nodes[0], nodes[1], Node.doubleHash(nodes[0].value + nodes[1].value)
            )

        left: Node = self.__buildTreeRec(nodes[:half])
        right: Node = self.__buildTreeRec(nodes[half:])
        value: str = Node.doubleHash(left.value + right.value)
        return Node(left, right, value)

    def printTree(self) -> None:
        self.__printTreeRec(self.root)

    def __printTreeRec(self, node) -> None:
        if node != None:
            print(node.value)
            self.__printTreeRec(node.left)
            self.__printTreeRec(node.right)

    def getRootHash(self) -> str:
        return self.root.value

```

1.  В нашем классе __Node__ каждый узел хранит свой хэш и ссылки на левый и 
    правый дочерние узлы. Мы добавим два статических метода для выполнения хэширования, используя алгоритм `SHA-256`. Для дополнительного уровня безопасности мы будем применять двойное хэширование.

    Учитывая структуру дерева __Меркла__, для построения нашего дерева нам необходимо четное количество конечных узлов. Если количество блоков данных нечетное, мы просто дважды хэшируем последний блок, создавая дубликат последнего конечного узла.

2.  Класс _Merkle_ включает в себя несколько методов:
    * Два метода для рекурсивного построения дерева.
    * Два метода для печати узлов в префиксном порядке.
    * Метод для получения корневого хэша.
    