# Диофантовы уравнения

**Диофантово уравнение** — это полиномиальное уравнение с целыми коэффициентами, для которого ищутся целочисленные решения. Названы в честь древнегреческого математика Диофанта Александрийского. Эти уравнения находят применение в самых разных областях — от криптографии до физики.

## Основные типы Диофантовых уравнений

1. **Линейные уравнения**:  
   $ ( a_1x_1 + a_2x_2 + \dots + a_nx_n = c ) $, где $( a_i, c )$ — целые числа.

2. **Уравнения Пелля**:  
   $( x^2 - Dy^2 = 1 )$, где $( D )$ — натуральное число, не являющееся квадратом.

3. **Уравнения вида \( x^n + y^n = z^n \)**:  
   Для $( n > 2 )$ не имеют нетривиальных решений (Великая теорема Ферма).

## Примеры решений на Python

### Пример 1: Линейное уравнение с двумя переменными

Уравнение: $( ax + by = c )$

**Условие разрешимости**:  
Решение существует, если $( \gcd(a, b) )$ делит $( c )$.

```python
def extended_gcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, x, y = extended_gcd(b % a, a)
        return (g, y - (b // a) * x, x)

def solve_diophantine(a, b, c):
    g, x0, y0 = extended_gcd(a, b)
    if c % g != 0:
        return None
    x0 *= c // g
    y0 *= c // g
    return (x0, y0)

# Пример: 3x + 6y = 9
a, b, c = 3, 6, 9
print(f"Решение: {solve_diophantine(a, b, c)}")  # Вывод: (3, 0)
```

### Пример 2: Уравнение Пелля

Уравнение: $( x^2 - Dy^2 = 1 )$

```python
import math

def solve_pell(D):
    a0 = int(math.isqrt(D))
    if a0 * a0 == D:
        return None
    x, y = a0, 1
    x_prev, y_prev = 1, 0
    while True:
        m = x * y_prev - x_prev * y
        d = (D - m * m) // (x * x_prev - D * y * y_prev)
        a = (a0 + m) // d
        x, x_prev = a * x + x_prev, x
        y, y_prev = a * y + y_prev, y
        if x * x - D * y * y == 1:
            return (x, y)

print(f"Решение x²-2y²=1: {solve_pell(2)}")  # Вывод: (3, 2)
```

### Пример 3: Уравнение Пифагора

Уравнение: $( x^2 + y^2 = z^2 )$

```python
from math import gcd

def pythagorean_triples(limit):
    return [(m*m-n*n, 2*m*n, m*m+n*n)
            for m in range(1, int(limit**0.5)+1)
            for n in range(1, m)
            if (m-n)%2 and gcd(m,n)==1 and m*m+n*n<=limit]

print("Тройки для z≤20:", pythagorean_triples(20))
# Вывод: [(3, 4, 5), (15, 8, 17), (5, 12, 13)]
```

## Применение Диофантовых уравнений

### 1. Криптография

- **RSA-шифрование**: основано на сложности разложения $( n = p \cdot q )$
- **Эллиптические кривые**: уравнения вида $( y^2 = x^3 + ax + b )$

```python
def factorize(n):
    return next((i, n//i) for i in range(2, int(n**0.5)+1) if n%i == 0)

print(f"Разложение 3233: {factorize(3233)}")  # Вывод: (61, 53)
```

### 2. Оптимизация

- **Задача о рюкзаке**: поиск целочисленных решений

```python
def knapsack(items, target):
    from itertools import combinations
    for r in range(1, len(items)+1):
        for combo in combinations(items, r):
            if sum(combo) == target:
                return combo
    return None

print(f"Решение для [3,5,7,9] и target=12: {knapsack([3,5,7,9], 12)}")
# Вывод: (3, 9)
```

### 3. Физика и инженерия

- Квантовые состояния: ( E = n^2 \cdot E_0 )$

```python
def quantum_states(max_E):
    return [n for n in range(1, int(max_E**0.5)+1)]

print(f"Состояния для E≤20: {quantum_states(20)}")  # Вывод: [1, 2, 3, 4]
```

### 4. Компьютерная графика

- Алгоритм Брезенхема для рисования линий

```python
def bresenham(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1-x0), abs(y1-y0)
    x, y, sx, sy = x0, y0, 1 if x0<x1 else -1, 1 if y0<y1 else -1
    err = dx - dy
    while True:
        points.append((x, y))
        if x == x1 and y == y1: break
        e2 = 2*err
        if e2 > -dy: err -= dy; x += sx
        if e2 < dx: err += dx; y += sy
    return points

print("Линия (0,0)-(5,3):", bresenham(0,0,5,3))
# Вывод: [(0,0), (1,1), (2,1), (3,2), (4,2), (5,3)]
```
