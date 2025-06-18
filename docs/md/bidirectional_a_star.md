# Двунаправленный поиск

Двунаправленный поиск представляет собой алгоритм поиска кратчайшего пути в ориентированном графе. Алгоритм выполняет два поиска одновременно: один — в прямом направлении от начальной вершины, другой — в обратном направлении от целевой вершины. Поиск останавливается, когда они пересекаются.
Такой подход позволяет ускорить процесс поиска, особенно в случаях, когда задача поиска упрощена. В такой модели сложности задачи поиска оба поиска расширяют дерево с коэффициентом ветвления __b__, а расстояние от начала до цели равно __d__. Каждый из двух поисков имеет сложность __O(bd/2)__, и сумма этих затрат времени на поиск меньше, чем сложность __O(bd)__, которая была бы получена в результате одного поиска от начала до цели.

Эндрю Голдберг и другие специалисты определили корректные условия завершения для двунаправленной версии алгоритма Дейкстры.

В алгоритме двунаправленного эвристического поиска, как и в алгоритме A*, используется эвристическая оценка оставшегося расстояния до цели (в прямом дереве) или от начала (в обратном дереве).

Алгоритм двунаправленного эвристического поиска был разработан и внедрён Айрой Полом. В отличие от алгоритма A*, деревья поиска, исходящие из начального и конечного узлов, не встречаются в середине пространства решений.

Алгоритм __BHFFA__ компании de Champeaux устраняет этот недостаток.
Решение, полученное с помощью однонаправленного алгоритма A* с использованием допустимой эвристики, является кратчайшим. Это свойство также характерно для двунаправленной эвристической версии алгоритма __BHFFA2__, предложенной де Шампо.

Алгоритм __BHFFA2__, помимо прочего, имеет более строгие условия завершения по сравнению с алгоритмом __BHFFA__.

## Реализация

```python title="python"
from __future__ import annotations

import time
from math import sqrt

# (1)!
HEURISTIC = 0

grid = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],  # (2)!
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0],
]

delta = [[-1, 0], [0, -1], [1, 0], [0, 1]]  # (3)!

TPosition = tuple[int, int]


class Node:
    def __init__(
        self,
        pos_x: int,
        pos_y: int,
        goal_x: int,
        goal_y: int,
        g_cost: int,
        parent: Node | None,
    ) -> None:
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos = (pos_y, pos_x)
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.g_cost = g_cost
        self.parent = parent
        self.h_cost = self.calculate_heuristic()
        self.f_cost = self.g_cost + self.h_cost

    def calculate_heuristic(self) -> float:
        # (4)!
        dy = self.pos_x - self.goal_x
        dx = self.pos_y - self.goal_y
        if HEURISTIC == 1:
            return abs(dx) + abs(dy)
        else:
            return sqrt(dy**2 + dx**2)

    def __lt__(self, other: Node) -> bool:
        return self.f_cost < other.f_cost


class AStar:
    def __init__(self, start: TPosition, goal: TPosition):
        self.start = Node(start[1], start[0], goal[1], goal[0], 0, None)
        self.target = Node(goal[1], goal[0], goal[1], goal[0], 99999, None)

        self.open_nodes = [self.start]
        self.closed_nodes: list[Node] = []

        self.reached = False

    def search(self) -> list[TPosition]:
        while self.open_nodes:
            # (5)!
            self.open_nodes.sort()
            current_node = self.open_nodes.pop(0)

            if current_node.pos == self.target.pos:
                return self.retrace_path(current_node)

            self.closed_nodes.append(current_node)
            successors = self.get_successors(current_node)

            for child_node in successors:
                if child_node in self.closed_nodes:
                    continue

                if child_node not in self.open_nodes:
                    self.open_nodes.append(child_node)
                else:
                    # (6)!
                    better_node = self.open_nodes.pop(self.open_nodes.index(child_node))

                    if child_node.g_cost < better_node.g_cost:
                        self.open_nodes.append(child_node)
                    else:
                        self.open_nodes.append(better_node)

        return [self.start.pos]

    def get_successors(self, parent: Node) -> list[Node]:
        # (7)!
        successors = []
        for action in delta:
            pos_x = parent.pos_x + action[1]
            pos_y = parent.pos_y + action[0]
            if not (0 <= pos_x <= len(grid[0]) - 1 and 0 <= pos_y <= len(grid) - 1):
                continue

            if grid[pos_y][pos_x] != 0:
                continue

            successors.append(
                Node(
                    pos_x,
                    pos_y,
                    self.target.pos_y,
                    self.target.pos_x,
                    parent.g_cost + 1,
                    parent,
                )
            )
        return successors

    def retrace_path(self, node: Node | None) -> list[TPosition]:
        # (8)!
        current_node = node
        path = []
        while current_node is not None:
            path.append((current_node.pos_y, current_node.pos_x))
            current_node = current_node.parent
        path.reverse()
        return path


class BidirectionalAStar:
    def __init__(self, start: TPosition, goal: TPosition) -> None:
        self.fwd_astar = AStar(start, goal)
        self.bwd_astar = AStar(goal, start)
        self.reached = False

    def search(self) -> list[TPosition]:
        while self.fwd_astar.open_nodes or self.bwd_astar.open_nodes:
            self.fwd_astar.open_nodes.sort()
            self.bwd_astar.open_nodes.sort()
            current_fwd_node = self.fwd_astar.open_nodes.pop(0)
            current_bwd_node = self.bwd_astar.open_nodes.pop(0)

            if current_bwd_node.pos == current_fwd_node.pos:
                return self.retrace_bidirectional_path(
                    current_fwd_node, current_bwd_node
                )

            self.fwd_astar.closed_nodes.append(current_fwd_node)
            self.bwd_astar.closed_nodes.append(current_bwd_node)

            self.fwd_astar.target = current_bwd_node
            self.bwd_astar.target = current_fwd_node

            successors = {
                self.fwd_astar: self.fwd_astar.get_successors(current_fwd_node),
                self.bwd_astar: self.bwd_astar.get_successors(current_bwd_node),
            }

            for astar in [self.fwd_astar, self.bwd_astar]:
                for child_node in successors[astar]:
                    if child_node in astar.closed_nodes:
                        continue

                    if child_node not in astar.open_nodes:
                        astar.open_nodes.append(child_node)
                    else:
                        # (9)!
                        better_node = astar.open_nodes.pop(
                            astar.open_nodes.index(child_node)
                        )

                        if child_node.g_cost < better_node.g_cost:
                            astar.open_nodes.append(child_node)
                        else:
                            astar.open_nodes.append(better_node)

        return [self.fwd_astar.start.pos]

    def retrace_bidirectional_path(
        self, fwd_node: Node, bwd_node: Node
    ) -> list[TPosition]:
        fwd_path = self.fwd_astar.retrace_path(fwd_node)
        bwd_path = self.bwd_astar.retrace_path(bwd_node)
        bwd_path.pop()
        bwd_path.reverse()
        path = fwd_path + bwd_path
        return path
```

1.  `1` для `manhattan`, `0` для `euclidean`

2.  `0` - свободный путь, тогда как `1` - препятствия

3.  Вверх, влево, вниз, вправо

4.  Эвристический подход для `A*`

5.  Открытые узлы сортируются с помощью __lt__

6.  Извлеките наилучший текущий путь

7.  Возвращает список последователей (как в сетке, так и в свободных местах)

8.  Проследите путь от родителей к родителям вплоть до начального узла

9.  Извлеките наилучший текущий путь
