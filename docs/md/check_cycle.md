---
tags:
  [
    Компьютерные науки,
    Базы данных,
    Экономика и финансы,
    Биотехнологии и медицина,
    Социальные сети,
  ]
---

# Проверка графа на наличие циклов

Для заданного графа необходимо определить, содержит ли он хотя бы один цикл. Следует отметить, что в рамках данной задачи не требуется найти все циклы, достаточно установить, существует ли хотя бы один. Поиск всех циклов является более сложной задачей.

## Основные бласти применения

1. Компьютерные науки (анализ зависимостей в системах сборки, например, Makefile)
2. Базы данных (обнаружение циклических зависимостей в транзакциях или схемах)
3. Экономика и финансы (выявление циклов в финансовых схемах или цепочках поставок)
4. Биотехнологии и медицина (исследование циклических взаимодействий в пищевых цепях или метаболических путях)
5. Социальные сети (анализ циклических связей в графах взаимодействий пользователей)

## Основные алгоритмы для проверки циклов

1. Поиск в глубину (DFS) – наиболее популярный метод.
2. Топологическая сортировка – если граф ациклический, его можно топологически отсортировать.
3. Алгоритм Union-Find (Disjoint Set Union, DSU) – эффективен для неориентированных графов.

## 1. Поиск в глубину (DFS) для ориентированного графа

Алгоритм:

- Помечаем вершины как `посещенные` (`visited`) и `в текущем рекурсивном обходе` (`rec_stack`).
- Если при обходе мы встречаем вершину, которая уже в `rec_stack`, значит, есть цикл.

### **Пример на Python**

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vertices = vertices

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def is_cyclic_util(self, v, visited, rec_stack):
        visited[v] = True
        rec_stack[v] = True

        for neighbor in self.graph[v]:
            if not visited[neighbor]:
                if self.is_cyclic_util(neighbor, visited, rec_stack):
                    return True
            elif rec_stack[neighbor]:
                return True

        rec_stack[v] = False
        return False

    def is_cyclic(self):
        visited = [False] * self.vertices
        rec_stack = [False] * self.vertices

        for node in range(self.vertices):
            if not visited[node]:
                if self.is_cyclic_util(node, visited, rec_stack):
                    return True
        return False

# Пример использования
g = Graph(4)
g.add_edge(0, 1)
g.add_edge(0, 2)
g.add_edge(1, 2)
g.add_edge(2, 0)  # <- цикл (2 → 0 → 1 → 2)
g.add_edge(2, 3)

if g.is_cyclic():
    print("Граф содержит цикл!")
else:
    print("Граф ациклический.")
```

**Вывод:**

```
Граф содержит цикл!
```

## 2. Поиск циклов в неориентированном графе (Union-Find)

Алгоритм:

- Используем структуру данных **Disjoint Set Union (DSU)**.
- Для каждого ребра проверяем, принадлежат ли его вершины одному множеству.
- Если да, то есть цикл.

### Пример на Python

```python
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return True  # Найден цикл!
        self.parent[root_x] = root_y
        return False

def has_cycle_undirected(edges, num_vertices):
    uf = UnionFind(num_vertices)
    for u, v in edges:
        if uf.union(u, v):
            return True
    return False

# Пример использования
edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # Цикл: 0 → 1 → 2 → 3 → 0
if has_cycle_undirected(edges, 4):
    print("Граф содержит цикл!")
else:
    print("Граф ациклический.")
```

**Вывод:**

```
Граф содержит цикл!
```

## 3. Топологическая сортировка (Kahn’s algorithm)

Если граф **ориентированный и ациклический (DAG)**, его можно отсортировать топологически. Если это не удается — есть цикл.

### **Пример на Python**

```python
from collections import deque

def is_cyclic_topological_sort(graph, num_vertices):
    in_degree = [0] * num_vertices
    for u in graph:
        for v in graph[u]:
            in_degree[v] += 1

    queue = deque([v for v in range(num_vertices) if in_degree[v] == 0])
    count = 0

    while queue:
        u = queue.popleft()
        count += 1
        for v in graph[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    return count != num_vertices  # Если count < num_vertices → есть цикл

# Пример использования
graph = {0: [1, 2], 1: [2], 2: [3], 3: [0]}  # Цикл: 0 → 1 → 2 → 3 → 0
if is_cyclic_topological_sort(graph, 4):
    print("Граф содержит цикл!")
else:
    print("Граф ациклический.")
```

**Вывод:**

```
Граф содержит цикл!
```
