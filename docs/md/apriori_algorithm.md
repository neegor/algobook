---
tags:
  [
    Торговля,
    Рекомендательные системы,
    Биотехнологии и медицина,
    Веб-аналитика,
    Экономика и финансы,
  ]
---

# Алгоритм Apriori

**Алгоритм Apriori** - это классический алгоритм для поиска ассоциативных правил в наборах данных, разработанный в 1994 году. Он широко используется в анализе рыночных корзин (`market basket analysis`) для выявления часто покупаемых вместе товаров.

## Основные бласти применения

1. Торговля (анализ рыночных корзин, выявление частых наборов товаров)
2. Рекомендательные системы (предложение сопутствующих товаров, как в Amazon)
3. Биотехнологии и медицина (анализ сочетаний симптомов и заболеваний)
4. Веб-аналитика (выявление паттернов поведения пользователей на сайтах)
5. Экономика и финансы (обнаружение мошеннических схем на основе транзакционных данных)

## Основные понятия

1. **Поддержка (Support)** - частота появления набора в данных  
   $ Support(X)$ = (Количество транзакций, содержащих $X$) / (Общее количество транзакций)

2. **Достоверность (Confidence)** - вероятность появления $Y$ при наличии $X$  
   $ Confidence(X → Y) = Support(X ∪ Y) / Support(X) $

3. **Лифт (Lift)** - насколько чаще встречается $Y$ вместе с $X$, чем ожидается  
   $ Lift(X → Y) = Confidence(X → Y) / Support(Y) $

## Принцип работы алгоритма Apriori

1. **Генерация кандидатов** - создание наборов элементов (itemsets) увеличенного размера
2. **Отсечение по поддержке** - удаление наборов, не удовлетворяющих минимальной поддержке
3. **Повторение** до тех пор, пока не перестанут генерироваться новые частые наборы

## Реализация на Python

### Пример 1: Простая реализация с нуля

```python
from itertools import combinations

def apriori(transactions, min_support):
    # Преобразование транзакций в множество элементов
    items = set()
    for transaction in transactions:
        for item in transaction:
            items.add(frozenset([item]))
    items = list(items)

    # Первый проход - вычисление поддержки для отдельных элементов
    item_counts = {}
    for item in items:
        for transaction in transactions:
            if item.issubset(transaction):
                item_counts[item] = item_counts.get(item, 0) + 1

    # Фильтрация по минимальной поддержке
    num_transactions = len(transactions)
    frequent_items = {}
    for item, count in item_counts.items():
        support = count / num_transactions
        if support >= min_support:
            frequent_items[item] = support

    # Генерация кандидатов большего размера
    k = 2
    current_frequent_items = frequent_items
    all_frequent_items = {}
    all_frequent_items.update(current_frequent_items)

    while current_frequent_items:
        # Генерация кандидатов
        itemsets = list(current_frequent_items.keys())
        candidates = set()
        for i in range(len(itemsets)):
            for j in range(i+1, len(itemsets)):
                candidate = itemsets[i].union(itemsets[j])
                if len(candidate) == k:
                    candidates.add(candidate)

        # Подсчет поддержки для кандидатов
        candidate_counts = {}
        for candidate in candidates:
            for transaction in transactions:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1

        # Фильтрация кандидатов
        current_frequent_items = {}
        for candidate, count in candidate_counts.items():
            support = count / num_transactions
            if support >= min_support:
                current_frequent_items[candidate] = support

        all_frequent_items.update(current_frequent_items)
        k += 1

    return all_frequent_items

# Пример данных
transactions = [
    ['молоко', 'хлеб', 'печенье'],
    ['молоко', 'печенье'],
    ['хлеб', 'печенье', 'кола'],
    ['хлеб', 'кола'],
    ['молоко', 'хлеб', 'печенье', 'кола'],
    ['молоко', 'хлеб', 'печенье']
]

# Запуск алгоритма
min_support = 0.5
frequent_itemsets = apriori(transactions, min_support)

# Вывод результатов
print("Частые наборы с поддержкой не менее", min_support)
for itemset, support in frequent_itemsets.items():
    print(f"{tuple(itemset)}: {support:.2f}")
```

### Пример 2: Использование библиотеки mlxtend

Более практичный способ - использовать готовую библиотеку:

```python
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Пример данных
dataset = [
    ['молоко', 'хлеб', 'печенье'],
    ['молоко', 'печенье'],
    ['хлеб', 'печенье', 'кола'],
    ['хлеб', 'кола'],
    ['молоко', 'хлеб', 'печенье', 'кола'],
    ['молоко', 'хлеб', 'печенье']
]

# Преобразование данных
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Нахождение частых наборов
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
print("Частые наборы:")
print(frequent_itemsets)

# Генерация ассоциативных правил
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("\nАссоциативные правила:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
```

### Пример 3: Визуализация результатов

```python
import matplotlib.pyplot as plt
import networkx as nx

# Создание графа
G = nx.DiGraph()

# Добавление узлов и ребер
for _, rule in rules.iterrows():
    G.add_edge(', '.join(rule['antecedents']),
               ', '.join(rule['consequents']),
               weight=rule['lift'])

# Рисование графа
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, k=0.5)
nx.draw(G, pos, with_labels=True,
        node_size=3000, node_color='skyblue',
        font_size=10, font_weight='bold',
        edge_color='gray', width=[d['weight']*0.5 for (u, v, d) in G.edges(data=True)])
edge_labels = {(u, v): f"Lift: {d['weight']:.2f}" for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Ассоциативные правила (размер стрелки соответствует лифту)")
plt.show()
```

## Оптимизации алгоритма Apriori

1. **Hash-based itemset counting** - использование хеш-таблиц для ускорения подсчета
2. **Transaction reduction** - удаление транзакций, не содержащих текущие частые наборы
3. **Partitioning** - разделение данных на части, которые можно обрабатывать в памяти
4. **Sampling** - работа с выборкой данных для начального анализа

## Ограничения алгоритма

1. Множественные проходы по данным
2. Генерация большого числа кандидатов
3. Высокие требования к памяти для больших наборов данных
4. Чувствительность к выбору минимальной поддержки

Алгоритм **Apriori** остается важной базовой техникой в анализе ассоциативных правил, несмотря на появление более современных алгоритмов, таких как **FP-Growth**.
