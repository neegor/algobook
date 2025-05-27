---
tags:
  - reqsys
---

# Policy Gradient (PG)

**Policy Gradient (PG)** — это семейство алгоритмов обучения с подкреплением (RL), которые **оптимизируют политику напрямую**, вместо того чтобы сначала оценивать Q-функцию (как в DQN). Это особенно полезно, когда:

- **Пространство действий большое или непрерывное** (например, рекомендация с плавным скором).
- **Нужна стохастическая политика** (например, для исследования новых рекомендаций).

## 1. Основная идея Policy Gradient

### Чем отличается от DQN?

| **Аспект**         | **DQN**                            | **Policy Gradient (PG)**          |
| ------------------ | ---------------------------------- | --------------------------------- |
| **Что обучается?** | Q-функция (ценность действий)      | Политика $\pi(a \| s)$ напрямую   |
| **Тип политики**   | Жёсткая (greedy/$\epsilon$-жадная) | Стохастическая (вероятностная)    |
| **Подходит для**   | Дискретные действия                | Дискретные и непрерывные действия |
| **Примеры**        | Deep Q-Learning                    | REINFORCE, Actor-Critic, PPO      |

### Формула градиента политики

Цель — максимизировать **ожидаемую награду** \( J(\theta) \):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s, a) \right]$$

где:

- $ \pi\_\theta(a|s) $ — вероятность выбора действия $ a $ в состоянии $ s $,
- $ Q^\pi(s, a) $ — ценность действия (можно аппроксимировать).

## 2. Алгоритмы Policy Gradient

### a) REINFORCE (Monte Carlo PG)

- Оценивает **полный возврат (return)** за эпизод.
- Прост в реализации, но имеет **высокую дисперсию**.

### b) Actor-Critic

- **Актор (Actor)** — выбирает действия (политика).
- **Критик (Critic)** — оценивает $ Q(s, a) $ или $ V(s) $.
- Снижает дисперсию по сравнению с REINFORCE.

### c) PPO (Proximal Policy Optimization)

- Современный алгоритм, который **ограничивает изменение политики** для стабильности.

## 3. Пример: REINFORCE для рекомендаций

### Задача

Рекомендовать один из 5 товаров пользователю на основе его истории.

### Код на PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.01, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))

    def learn(self):
        returns = []
        G = 0
        # Рассчитываем дисконтированные возвраты с конца эпизода
        for reward in reversed([x[2] for x in self.memory]):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)  # Нормализация

        policy_loss = []
        for (state, action, _), G in zip(self.memory, returns):
            state = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy(state)
            log_prob = torch.log(probs.squeeze(0)[action])
            policy_loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.memory = []  # Очищаем память после обучения

# Пример использования
state_dim = 10  # Размерность состояния (например, эмбеддинг пользователя)
action_dim = 5  # 5 товаров для рекомендации
agent = REINFORCE(state_dim, action_dim)

# Имитация одного эпизода (пользовательский сеанс)
states = [np.random.randn(state_dim) for _ in range(10)]  # 10 шагов
rewards = [np.random.choice([-0.1, 0.5, 1.0]) for _ in range(10)]  # Случайные награды

for state, reward in zip(states, rewards):
    action = agent.act(state)
    agent.remember(state, action, reward)

agent.learn()  # Обновляем политику
```

## 4. Пример: Actor-Critic для рекомендаций

### Код

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ActorCritic:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        self.gamma = gamma
        self.memory = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.actor(state)
        action = torch.multinomial(probs, 1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        states, actions, rewards, next_states, dones = zip(*self.memory)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Критик оценивает V(s)
        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze().detach()
        targets = rewards + (1 - dones) * self.gamma * next_values
        td_errors = targets - values

        # Обновляем актора
        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze())
        actor_loss = -(log_probs * td_errors).mean()

        # Обновляем критика
        critic_loss = nn.MSELoss()(values, targets)

        # Общий loss
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []
```

## 5. Проблемы Policy Gradient и решения

### a) Высокая дисперси

- **Решение**: Использовать **базовую линию (baseline)** или **Actor-Critic**.

### **b) Неэффективное исследование**

- **Решение**: **Энтропийная регуляризация** (поощряет разнообразие действий).

### **c) Нестабильность обучения**

- **Решение**: **PPO** или **TRPO** (ограничивают изменение политики).

## 6. Где применяется?

- **Spotify** → Для адаптации плейлистов.
- **Netflix** → Персонализация рекомендаций.
- **Доставка еды** → Оптимизация порядка ресторанов в ленте.

## 7. Будущее PG в рекомендациях

- **PPO + Трансформеры** → Учёт долгосрочной истории пользователя.
- **Мета-обучение** → Быстрая адаптация к новым пользователям.
