---
tags:
  [
    Игровая разработка,
    Робототехника,
    Экономика и финансы,
    Биотехнологии и медицина,
    Научные вычисления,
    Машинное обучение,
  ]
---

# SAC-RND

SAC-RND (Soft Actor-Critic with Random Network Distillation) - это модификация алгоритма Soft Actor-Critic, которая добавляет механизм внутренней мотивации через технику Random Network Distillation (RND) для улучшения исследования в средах с редкими вознаграждениями. SAC-RND и подобные алгоритмы, сочетающие максимальную энтропию (SAC) с методами внутренней мотивации (RND), находят применение в областях, где требуется **автономное исследование среды** и **обучение в условиях редких или отсутствующих внешних вознаграждений**.

## Основные бласти применения

1. Робототехника (автономное обучение роботов и исследование неизвестных сред)
2. Игровая разработка (обучение с нуля (без экспертных данных и генерация разнообразного поведения)
3. Машинное обучение (персонализированные рекомендации)
4. Научные вычисления (изучение физических/химических процессов)
5. Биотехнологии и медицина (анализ белковых структур)
6. Экономика и финансы (обучение торговых агентов)

## Основные компоненты SAC-RND

1. **Soft Actor-Critic (SAC)**: Алгоритм актор-критик с максимальной энтропией, который максимизирует не только награду, но и стохастичность политики.

2. **Random Network Distillation (RND)**: Метод исследования, где:
   - Фиксированная случайная нейронная сеть (target) генерирует "интересные" признаки состояний
   - Вторая сеть (predictor) обучается предсказывать выход target сети
   - Ошибка предсказания используется как внутренняя награда за исследование

## Принцип работы SAC-RND

1. Для каждого состояния вычисляется ошибка предсказания RND
2. Эта ошибка добавляется к внешней награде как внутренняя мотивация
3. Агент получает бонус за посещение состояний, которые плохо предсказываются

## Реализация на Python

Вот пример реализации SAC-RND с использованием PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sac import SAC  # Предполагаем, что у нас есть базовая реализация SAC

class RNDNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(RNDNetwork, self).__init__()
        # Target network (fixed)
        self.target = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)

        # Predictor network (trainable)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, state):
        with torch.no_grad():
            target_features = self.target(state)
        predicted_features = self.predictor(state)
        return target_features, predicted_features

class SAC_RND(SAC):
    def __init__(self, state_dim, action_dim, rnd_scale=0.1, **kwargs):
        super(SAC_RND, self).__init__(state_dim, action_dim, **kwargs)
        self.rnd = RNDNetwork(state_dim)
        self.rnd_optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=1e-4)
        self.rnd_scale = rnd_scale

    def compute_intrinsic_reward(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        target, predicted = self.rnd(state)
        intrinsic_reward = torch.mean((target - predicted)**2, dim=1).item()
        return intrinsic_reward * self.rnd_scale

    def update_rnd(self, batch):
        states = torch.FloatTensor(batch['states']).to(self.device)
        target, predicted = self.rnd(states)

        # Update predictor
        rnd_loss = torch.mean((target.detach() - predicted)**2)
        self.rnd_optimizer.zero_grad()
        rnd_loss.backward()
        self.rnd_optimizer.step()

        return rnd_loss.item()

    def train(self, batch):
        # Обновляем RND
        rnd_loss = self.update_rnd(batch)

        # Вычисляем внутреннюю награду для всех состояний в батче
        intrinsic_rewards = np.array([self.compute_intrinsic_reward(s) for s in batch['states']])
        batch['rewards'] += intrinsic_rewards

        # Обычное обновление SAC
        sac_losses = super().train(batch)

        return {**sac_losses, 'rnd_loss': rnd_loss}
```

## Пример использования

```python
import gym
from collections import deque
import random

env = gym.make('MountainCarContinuous-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SAC_RND(state_dim, action_dim, rnd_scale=0.1)

episodes = 1000
batch_size = 256
replay_buffer = deque(maxlen=100000)

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # Сохраняем переход в буфере
        replay_buffer.append((state, action, reward, next_state, done))
        episode_reward += reward
        state = next_state

        # Обучение, когда набрано достаточно примеров
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            batch = {
                'states': np.array([x[0] for x in batch]),
                'actions': np.array([x[1] for x in batch]),
                'rewards': np.array([x[2] for x in batch]),
                'next_states': np.array([x[3] for x in batch]),
                'dones': np.array([x[4] for x in batch])
            }
            losses = agent.train(batch)

    print(f"Episode {episode}, Reward: {episode_reward}, RND Loss: {losses.get('rnd_loss', 0)}")
```

## Ключевые особенности реализации

1. **Две сети в RND**: Target сеть фиксирована и служит эталоном, predictor сеть обучается предсказывать её выход.

2. **Внутренняя награда**: Вычисляется как MSE между выходами target и predictor сетей.

3. **Масштабирование**: Параметр `rnd_scale` контролирует влияние внутренней награды.

4. **Совместное обучение**: RND обновляется вместе с основным алгоритмом SAC.

## Преимущества SAC-RND

1. Улучшенное исследование в средах с редкими наградами
2. Автоматическая адаптация уровня исследования
3. Сохранение преимуществ SAC (эффективность, стабильность)
4. Не требует предварительных знаний о среде

SAC-RND особенно полезен в задачах, где внешние награды редки или отсутствуют, таких как исследование неизвестных сред или обучение с подкреплением с нуля.
