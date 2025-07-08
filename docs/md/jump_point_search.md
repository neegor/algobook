---
tags: [Игровая разработка, Робототехника, Логистика, ГИС и картография]
---

# Jump Point Search (JPS)

**Jump Point Search (JPS)** - это оптимизированный алгоритм поиска пути на сетках, который является специализированной версией **A\***. Он особенно эффективен для uniform-cost grid environments (сеток с одинаковой стоимостью перемещения между клетками).

## Основные бласти применения

1. Игровая разработка (поиск пути для NPC в стратегиях или RPG с крупными картами, навигация агентов в симулированных пространствах)
2. Робототехника (планирование траектории мобильных роботов в статичной среде)
3. Логистика (оптимизация маршрутов в логистике и навигационных приложениях)
4. ГИС и картография (построение кратчайших путей в сеточных представлениях местности)

## Основные принципы Jump Point Search

1. **Исключение симметричных путей**: JPS избегает обработки избыточных путей, которые приводят к одной и той же точке с той же стоимостью.

2. **Прыжки (Jumping)**: Вместо проверки каждого соседа, алгоритм "прыгает" через клетки, пока не найдет "точку перехода" (jump point) - место, где путь отклоняется от прямой.

3. **Направленные правила**: Алгоритм использует правила принудительных соседей (forced neighbors) для определения ключевых точек.

## Преимущества перед A\*

- **Быстрее**: В типичных случаях работает в 10-100 раз быстрее A\*
- **Меньше памяти**: Открытый список содержит значительно меньше узлов
- **Тот же результат**: Находит оптимальный путь, как и A\*

## Пример реализации JPS

```python
import heapq
from typing import List, Tuple, Dict, Optional, Set

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Эвристическая функция (расстояние Чебышева)"""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return dx + dy + (1.414 - 2) * min(dx, dy)

class Node:
    def __init__(self, position: Tuple[int, int], parent: Optional['Node'] = None):
        self.position = position
        self.parent = parent
        self.g = 0  # стоимость от начала
        self.h = 0  # эвристическая оценка
        self.f = 0  # суммарная стоимость

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

    def __hash__(self):
        return hash(self.position)

def is_valid(grid: List[List[int]], pos: Tuple[int, int]) -> bool:
    """Проверяет, находится ли позиция в пределах сетки и проходима ли она"""
    x, y = pos
    return 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0

def has_forced_neighbor(grid: List[List[int]], x: int, y: int, dx: int, dy: int) -> bool:
    """Проверяет наличие принудительных соседей"""
    if dx != 0 and dy != 0:  # диагональное движение
        # Горизонтальные принудительные соседи
        if (is_valid(grid, (x, y - dy)) and not is_valid(grid, (x - dx, y - dy)) or
            is_valid(grid, (x, y + dy)) and not is_valid(grid, (x - dx, y + dy))):
            return True
        # Вертикальные принудительные соседи
        if (is_valid(grid, (x - dx, y)) and not is_valid(grid, (x - dx, y - dy)) or
            is_valid(grid, (x + dx, y)) and not is_valid(grid, (x + dx, y - dy))):
            return True
    else:  # ортогональное движение
        if dx == 0:  # вертикальное движение
            if (is_valid(grid, (x + 1, y + dy)) and not is_valid(grid, (x + 1, y)) or
                is_valid(grid, (x - 1, y + dy)) and not is_valid(grid, (x - 1, y))):
                return True
        else:  # горизонтальное движение
            if (is_valid(grid, (x + dx, y + 1)) and not is_valid(grid, (x, y + 1)) or
                is_valid(grid, (x + dx, y - 1)) and not is_valid(grid, (x, y - 1))):
                return True
    return False

def jump(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int],
          dx: int, dy: int) -> Optional[Tuple[int, int]]:
    """Функция прыжка: ищет следующую точку перехода"""
    x, y = start
    while True:
        x += dx
        y += dy

        # Если достигли конечной точки
        if (x, y) == end:
            return (x, y)

        # Если клетка непроходима или вне сетки
        if not is_valid(grid, (x, y)):
            return None

        # Проверка принудительных соседей
        if has_forced_neighbor(grid, x, y, dx, dy):
            return (x, y)

        # Рекурсивные прыжки по диагонали
        if dx != 0 and dy != 0:
            # Горизонтальный прыжок
            if jump(grid, (x, y), end, dx, 0) is not None:
                return (x, y)
            # Вертикальный прыжок
            if jump(grid, (x, y), end, 0, dy) is not None:
                return (x, y)

    return None

def get_successors(grid: List[List[int]], node: Node, end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Возвращает список точек перехода для данного узла"""
    successors = []
    x, y = node.position

    # Получаем направления движения от родителя
    if node.parent is None:
        # Начальная точка - проверяем все направления
        directions = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    else:
        # Определяем естественное направление движения
        px, py = node.parent.position
        dx, dy = x - px, y - py
        dx = max(-1, min(1, dx))
        dy = max(-1, min(1, dy))

        # Базовые направления
        if dx != 0 and dy != 0:  # диагональное движение
            directions = [
                (dx, dy),  # продолжение диагонали
                (dx, 0),   # горизонтальное
                (0, dy)    # вертикальное
            ]
        else:  # ортогональное движение
            if dx == 0:  # вертикальное
                directions = [
                    (0, dy),  # продолжение вертикали
                    (1, dy),  # диагонали
                    (-1, dy)
                ]
            else:  # горизонтальное
                directions = [
                    (dx, 0),  # продолжение горизонтали
                    (dx, 1),  # диагонали
                    (dx, -1)
                ]

    # Проверяем каждое направление на наличие точек перехода
    for dx, dy in directions:
        jx, jy = x + dx, y + dy
        if is_valid(grid, (jx, jy)):
            jump_point = jump(grid, (x, y), end, dx, dy)
            if jump_point is not None:
                successors.append(jump_point)

    return successors

def jps(grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Реализация алгоритма Jump Point Search"""
    # Инициализация начального и конечного узлов
    start_node = Node(start)
    end_node = Node(end)

    # Открытый и закрытый списки
    open_list = []
    closed_list = set()

    # Добавляем начальный узел
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        # Если достигли цели
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Разворачиваем путь

        closed_list.add(current_node.position)

        # Получаем точки перехода
        successors = get_successors(grid, current_node, end)

        for successor in successors:
            if successor in closed_list:
                continue

            # Создаем новый узел
            new_node = Node(successor, current_node)

            # Вычисляем стоимости
            dx = new_node.position[0] - current_node.position[0]
            dy = new_node.position[1] - current_node.position[1]
            step_cost = 1 if dx == 0 or dy == 0 else 1.414  # стоимость диагонального шага

            new_node.g = current_node.g + step_cost
            new_node.h = heuristic(new_node.position, end)
            new_node.f = new_node.g + new_node.h

            # Проверяем, есть ли уже такой узел в открытом списке с меньшей стоимостью
            found = False
            for node in open_list:
                if node == new_node and node.g <= new_node.g:
                    found = True
                    break

            if not found:
                heapq.heappush(open_list, new_node)

    return []  # Путь не найден

# Пример использования
if __name__ == "__main__":
    # 0 - проходимая клетка, 1 - препятствие
    grid = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    start = (0, 0)
    end = (7, 6)

    path = jps(grid, start, end)
    print("Найденный путь JPS:")
    for position in path:
        print(position)
```

## Визуализация пути JPS

```python
def print_grid_with_path(grid, path):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if (i, j) == path[0]:
                print("S", end=" ")  # Start
            elif (i, j) == path[-1]:
                print("E", end=" ")  # End
            elif (i, j) in path:
                print("*", end=" ")  # Path
            else:
                print(grid[i][j], end=" ")
        print()

print("\nВизуализация пути JPS:")
print_grid_with_path(grid, path)
```

## Ключевые особенности реализации

1. **Функция jump()**: Рекурсивно ищет следующую точку перехода в заданном направлении.

2. **Правила принудительных соседей**: Функция `has_forced_neighbor` определяет, есть ли соседи, которые делают текущую клетку точкой перехода.

3. **Диагональные движения**: Учитываются с соответствующей стоимостью (√2 ≈ 1.414).

4. **Оптимизация направлений**: При определении преемников учитывается направление движения от родительского узла.

## Сравнение JPS и A\*

1. **Производительность**: JPS обычно быстрее, особенно на больших открытых пространствах.
2. **Память**: JPS хранит меньше узлов в открытом списке.
3. **Сложность реализации**: JPS сложнее в реализации из-за правил прыжков и принудительных соседей.
4. **Применимость**: JPS работает только на uniform grids, тогда как A\* более универсален.

Jump Point Search особенно полезен в играх и приложениях, где требуется частый поиск пути на больших сетках с однородной стоимостью перемещения.
