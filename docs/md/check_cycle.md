# Проверка графа на наличие циклов

Для заданного графа необходимо определить, содержит ли он хотя бы один цикл. Следует отметить, что в рамках данной задачи не требуется найти все циклы, достаточно установить, существует ли хотя бы один. Поиск всех циклов является более сложной задачей.

## Реализация

```python title="python"
def check_cycle(graph: dict) -> bool:
    # (1)!
    visited: set[int] = set()
    # (2)!
    rec_stk: set[int] = set()
    return any(
        node not in visited and depth_first_search(graph, node, visited, rec_stk)
        for node in graph
    )


def depth_first_search(graph: dict, vertex: int, visited: set, rec_stk: set) -> bool:
    # (3)!
    visited.add(vertex)
    rec_stk.add(vertex)

    for node in graph[vertex]:
        if node not in visited:
            if depth_first_search(graph, node, visited, rec_stk):
                return True
        elif node in rec_stk:
            return True

    # (4)!
    rec_stk.remove(vertex)
    return False
```

1.  Следите за посещенными узлами

2.  Чтобы обнаружить обратное ребро, следите за вершинами, которые в данный момент находятся в стеке рекурсии

3.  Отметьте текущий узел как посещенный и добавьте в стек рекурсии

4.  Узел должен быть исключен из стека рекурсии до выхода из функции
