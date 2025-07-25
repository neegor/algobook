---
tags:
  [
    Игровая разработка,
    Робототехника,
    Экономика и финансы,
    Биотехнологии и медицина,
    Научные вычисления,
    Машинное обучение,
    Автономные транспортные средства,
  ]
---

# Random Network Distillation (RND)

Random Network Distillation (RND) - это метод исследования в reinforcement learning (RL), который помогает агенту изучать среду через внутреннее вознаграждение за новизну. Он был представлен исследователями из OpenAI в 2018 году.

## Основные области применения

**Обучение с подкреплением (Reinforcement Learning):**

- Исследование среды (RND помогает агентам изучать новые состояния в средах с разреженными наградами (sparse rewards), например)
- Игры (для открытия новых стратегий)
- Робототехника (обучение роботов сложным движениям без явных reward-функций)
- Долгосрочные задачи (В задачах, где награда отложена во времени (например, сбор редких предметов))

**Автономные системы и робототехника**

- Обучение роботов
- Беспилотные автомобили

## Основная идея RND

RND состоит из двух нейронных сетей:

1. **Целевая сеть (target network)** - случайно инициализированная и замороженная сеть
2. **Предсказывающая сеть (predictor network)** - обучается предсказывать выход целевой сети

Когда агент встречает новое состояние, predictor делает плохие предсказания, что создает высокую ошибку предсказания (внутреннее вознаграждение). По мере того как состояние становится более знакомым, ошибка уменьшается.

## Применение RND

RND особенно полезен в:

- Разведывательных задачах (exploration)
- Средах с редкими или отсутствующими внешними вознаграждениями
- Обучении с подкреплением в сложных средах

## Реализация на Python

Вот пример реализации RND с использованием PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RNDNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RNDNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

class RND:
    def __init__(self, input_dim, lr=1e-4, device='cpu'):
        self.device = device
        self.target = RNDNetwork(input_dim).to(device)
        self.predictor = RNDNetwork(input_dim).to(device)

        # Замораживаем веса целевой сети
        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def compute_intrinsic_reward(self, state):
        """
        Вычисляет внутреннее вознаграждение на основе ошибки предсказания
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            target_features = self.target(state_tensor)

        predicted_features = self.predictor(state_tensor)
        reward = torch.mean((predicted_features - target_features)**2, dim=1)
        return reward.cpu().detach().numpy()

    def update(self, states):
        """
        Обновляет предсказывающую сеть
        """
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)

        # Получаем целевые фичи
        with torch.no_grad():
            target_features = self.target(states_tensor)

        # Предсказываем фичи
        predicted_features = self.predictor(states_tensor)

        # Вычисляем потери и обновляем
        loss = self.loss_fn(predicted_features, target_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

# Пример использования
if __name__ == "__main__":
    # Параметры
    state_dim = 10  # Размерность состояния
    batch_size = 32

    # Инициализация RND
    rnd = RND(state_dim)

    # Генерация случайных состояний
    states = np.random.randn(batch_size, state_dim)

    # Вычисление внутреннего вознаграждения
    intrinsic_rewards = rnd.compute_intrinsic_reward(states)
    print("Intrinsic rewards for new states:", intrinsic_rewards)

    # Обучение на этих состояниях
    for _ in range(100):
        loss = rnd.update(states)

    # Вычисление вознаграждения после обучения
    intrinsic_rewards_after = rnd.compute_intrinsic_reward(states)
    print("Intrinsic rewards after training:", intrinsic_rewards_after)
```

## Интеграция RND с алгоритмом обучения с подкреплением

Вот как можно интегрировать RND с алгоритмом PPO:

```python
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

class RNDPPO(PPO):
    def __init__(self, *args, rnd_kwargs=None, intrinsic_weight=1.0, **kwargs):
        super(RNDPPO, self).__init__(*args, **kwargs)
        self.rnd = RND(self.observation_space.shape[0], **rnd_kwargs or {})
        self.intrinsic_weight = intrinsic_weight

    def collect_rollouts(self, *args, **kwargs):
        # Переопределяем сбор rollout для добавления внутреннего вознаграждения
        rollout_data = super().collect_rollouts(*args, **kwargs)

        # Вычисляем внутреннее вознаграждение
        intrinsic_rewards = self.rnd.compute_intrinsic_reward(rollout_data.observations)

        # Комбинируем внешнее и внутреннее вознаграждение
        rollout_data.rewards += self.intrinsic_weight * intrinsic_rewards

        # Обновляем RND
        self.rnd.update(rollout_data.observations)

        return rollout_data

# Пример использования с окружением
env = DummyVecEnv([lambda: gym.make("MountainCarContinuous-v0")])

model = RNDPPO(
    "MlpPolicy",
    env,
    rnd_kwargs={"hidden_dim": 64, "lr": 1e-4},
    intrinsic_weight=0.1,
    verbose=1
)

model.learn(total_timesteps=10000)
```

## Важные аспекты RND

1. **Инициализация целевой сети**: Целевая сеть инициализируется случайно и никогда не обучается.
2. **Обучение предсказывающей сети**: Предсказывающая сеть обучается минимизировать MSE между своим выходом и выходом целевой сети.
3. **Внутреннее вознаграждение**: Ошибка предсказания используется как мера новизны состояния.
4. **Гибридное вознаграждение**: В реальных задачах часто комбинируют внешнее и внутреннее вознаграждение.

## Преимущества RND

- Не требует хранения предыдущих состояний (в отличие от count-based методов)
- Масштабируется на высокоразмерные пространства состояний
- Эффективен в средах с редкими вознаграждениями

RND стал важным компонентом многих современных алгоритмов обучения с подкреплением, особенно в задачах, где исследование среды критически важно.
