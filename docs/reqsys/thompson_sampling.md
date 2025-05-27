---
tags:
  - reqsys
---

# Thompson Sampling

Thompson Sampling (TS) — это один из самых эффективных алгоритмов для решения проблемы многорукого бандита, который особенно хорошо подходит для рекомендательных систем:

- В условиях неопределённости
- При необходимости автоматического баланса между исследованием и эксплуатацией
- В сценариях с ограниченными данными.

## Основная идея Thompson Sampling

Thompson Sampling — это байесовский подход, который:

1. Моделирует неопределённость в оценках наград для каждого действия
2. Выбирает действия путём выборки из апостериорного распределения
3. Обновляет свои представления на основе полученных наград

### Математическая основа

Для бинарных наград (клик/не клик) TS часто использует бета-распределение:

- Для каждого действия a поддерживается Beta(αₐ, βₐ)
- αₐ — количество успехов (кликов)
- βₐ — количество неудач (пропусков)

Алгоритм:

1. Для каждого действия a делаем выборку θₐ ~ Beta(αₐ, βₐ)
2. Выбираем действие с максимальным θₐ
3. Получаем награду r (0 или 1)
4. Обновляем параметры:
   - Если r=1: αₐ ← αₐ + 1
   - Если r=0: βₐ ← βₐ + 1

## Преимущества Thompson Sampling

1. **Автоматический баланс exploration/exploitation**: больше исследует действия с высокой неопределённостью
2. **Вычислительная эффективность**: требует только выборки из распределения
3. **Хорошие теоретические гарантии**: достигает логарифмического сожаления
4. **Простота реализации**

## Пример реализации на Python

Рассмотрим задачу рекомендации новостей, где нужно выбирать из нескольких вариантов заголовков.

```python
import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

class ThompsonSampling:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms)  # Количество успехов для каждого действия
        self.beta = np.ones(n_arms)   # Количество неудач для каждого действия

    def select_arm(self):
        # Выборка из бета-распределения для каждого действия
        samples = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        # Обновление параметров выбранного действия
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += (1 - reward)

    def plot_distributions(self):
        x = np.linspace(0, 1, 100)
        for arm in range(self.n_arms):
            y = beta.pdf(x, self.alpha[arm], self.beta[arm])
            plt.plot(x, y, label=f'Action {arm}')
        plt.legend()
        plt.title('Posterior Distributions')
        plt.show()
```

## Пример использования

```python
# Параметры симуляции
n_arms = 3  # Количество вариантов новостей
true_means = [0.3, 0.5, 0.7]  # Истинные вероятности клика для каждого варианта
n_trials = 1000  # Количество показов

# Инициализация
ts = ThompsonSampling(n_arms)
rewards = []

for t in range(n_trials):
    # Выбор действия
    chosen_arm = ts.select_arm()

    # Симуляция награды (клик или нет)
    reward = np.random.binomial(1, true_means[chosen_arm])

    # Обновление модели
    ts.update(chosen_arm, reward)
    rewards.append(reward)

    # Визуализация распределений каждые 200 шагов
    if t in [50, 200, 500, 999]:
        print(f"Step {t}: Alpha = {ts.alpha}, Beta = {ts.beta}")
        ts.plot_distributions()

# Вычисление кумулятивного сожаления
optimal_rewards = [max(true_means) for _ in range(n_trials)]
cumulative_regret = np.cumsum(optimal_rewards - np.array(rewards))

plt.plot(cumulative_regret)
plt.xlabel('Trials')
plt.ylabel('Cumulative Regret')
plt.title('Performance of Thompson Sampling')
plt.show()
```

## Анализ результатов

1. **Распределения вероятностей**:

   - В начале все распределения одинаковы (равномерные)
   - Со временем распределения для лучших действий становятся уже и смещаются вправо
   - Для плохих действий распределения остаются широкими или смещаются влево

2. **Кумулятивное сожаление**:
   - В идеале должно расти логарифмически
   - Быстрое схождение к оптимальному действию

## Расширения Thompson Sampling

### 1. Контекстный Thompson Sampling (Linear Thompson Sampling)

Для учёта контекста пользователя можно использовать линейную модель:

```python
class LinearThompsonSampling:
    def __init__(self, n_arms, context_dim, lambda_=1.0):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.lambda_ = lambda_

        # Инициализация параметров
        self.B = [np.identity(context_dim) for _ in range(n_arms)]
        self.mu = [np.zeros(context_dim) for _ in range(n_arms)]
        self.f = [np.zeros(context_dim) for _ in range(n_arms)]

    def select_arm(self, context):
        samples = []
        for a in range(self.n_arms):
            # Выборка из многомерного нормального распределения
            mu_hat = self.mu[a]
            B_inv = np.linalg.inv(self.B[a])
            sample = np.random.multivariate_normal(mu_hat, B_inv)
            samples.append(np.dot(sample, context))

        return np.argmax(samples)

    def update(self, chosen_arm, context, reward):
        # Обновление параметров
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu[chosen_arm] = np.linalg.solve(self.B[chosen_arm], self.f[chosen_arm])
```

### 2. Нелинейный Thompson Sampling

Для сложных зависимостей можно использовать нейронные сети с dropout в качестве аппроксимации байесовской нейросети.

## Практические советы

1. **Инициализация**:

   - Можно начинать с α=1, β=1 (равномерное распределение)
   - Для "тёплого старта" можно использовать исторические данные

2. **Непрерывные награды**:

   - Для непрерывных наград (например, время чтения) используйте нормальное распределение вместо бета

3. **Масштабирование**:
   - Для больших каталогов можно использовать кластеризацию действий
   - Или применять методы уменьшения размерности для контекста

## Сравнение с другими методами

| Метод             | Плюсы                                                      | Минусы                                  |
| ----------------- | ---------------------------------------------------------- | --------------------------------------- |
| Thompson Sampling | Автоматический exploration, хорошие теоретические гарантии | Требует байесовского обновления         |
| LinUCB            | Прозрачность, хорошая интерпретируемость                   | Жёсткий баланс exploration/exploitation |
| ε-жадный          | Простота реализации                                        | Неэффективное исследование              |

