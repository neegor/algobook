# Deep Q-Networks (DQN)

**Deep Q-Networks (DQN)** — это гибрид глубокого обучения и Q-learning, который стал прорывом в применении RL для рекомендательных систем. Давайте разберём его принципы, архитектуру и реализацию на практике.

## 1. Что такое Deep Q-Networks (DQN)?

DQN — это метод **обучения с подкреплением (RL)**, который использует нейронную сеть для аппроксимации **Q-функции** (функции ценности действий).

### Ключевые идеи

- **Q-learning** → Оценка "полезности" действия в состоянии:

  $$Q(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q(s', a')]$$

  где:

  - $ s $ — текущее состояние,
  - $ a $ — действие (рекомендация),
  - $ r $ — награда (клик, покупка),
  - $ \gamma $ — коэффициент дисконтирования.

- **Нейросеть вместо таблицы Q-values** → Позволяет работать с большими пространствами состояний (например, история просмотров пользователя).

## 2. Архитектура DQN

### Основные компоненты

1. **Входной слой** → Вектор состояния (например, эмбеддинг пользователя + история действий).
2. **Скрытые слои** → Полносвязные или CNN/LSTM для обработки сложных данных.
3. **Выходной слой** → Q-значения для каждого возможного действия (например, каждого товара в каталоге).

### Два ключевых усовершенствования

- **Experience Replay** → Буфер памяти для хранения прошлых переходов $(s, a, r, s')$. Обучение идёт на случайных батчах из буфера, чтобы избежать корреляции между последовательными состояниями.
- **Target Network** → Отдельная сеть для стабильного расчёта целевых Q-значений (обновляется периодически).

## 3. Пример DQN для рекомендаций

### Задача

Рекомендовать фильмы пользователю на основе его истории просмотров.

### **Данные**

- **Состояние (State)** → Эмбеддинг пользователя + последние 5 просмотренных фильмов.
- **Действие (Action)** → Выбор одного из 10 фильмов для рекомендации.
- **Награда (Reward)** →
  - +1 если пользователь кликнул,
  - +10 если посмотрел до конца,
  - -0.1 если проигнорировал.

### Код на PyTorch

```python
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Пример использования
state_dim = 100  # Размерность эмбеддинга состояния
action_dim = 10  # 10 возможных фильмов для рекомендации
agent = DQNAgent(state_dim, action_dim)

# Цикл обучения
for episode in range(1000):
    state = np.random.randn(state_dim)  # Имитация состояния пользователя
    for step in range(100):
        action = agent.act(state)
        next_state = np.random.randn(state_dim)  # Новое состояние
        reward = np.random.choice([-0.1, 1, 10])  # Случайная награда
        done = step == 99
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.replay(32)  # Обучение на батче из 32 примеров
    if episode % 10 == 0:
        agent.update_target_model()
```

---

## 4. Проблемы DQN и их решения

### a. Переоценка Q-значений

- **Проблема**: DQN склонен завышать оценки.
- **Решение**: Double DQN (разделение выбора действия и оценки).

### b. Неэффективность для больших каталогов

- **Проблема**: Если действий тысячи (например, все товары Amazon), выходной слой слишком большой.
- **Решение**:
  - **Action Embeddings** → Сводят действия в низкоразмерное пространство.
  - **DQN с вниманием** → Например, **DRN** (Deep Reinforcement Learning for Recommendations).

### c. Холодный старт

- **Решение**: Предобучение на имитационных данных или гибрид с collaborative filtering.

## 5. Где применяется DQN в рекомендациях?

- **YouTube** (ранние версии рекомендаций).
- **Alibaba** → Для динамического ретаргетинга товаров.
- **Новостные агрегаторы** → Персонализация ленты.

## 6. Будущее DQN

- **Комбинация с трансформерами** → Например, **Decision Transformer** для рекомендаций.
- **Мета-обучение** → Быстрая адаптация к новым пользователям.
