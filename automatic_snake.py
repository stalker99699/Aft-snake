import pygame
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ... (остальные импорты и константы из предыдущего кода)

class DQN(nn.Module):
    """Нейронная сеть для принятия решений"""
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class RLAgent:
    """Агент с машинным обучением"""
    def __init__(self, input_size, output_size):
        # Параметры обучения
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        # Модели
        self.model = DQN(input_size, 128, output_size)
        self.target_model = DQN(input_size, 128, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.update_target_model()
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_state(self, game):
        """Получение текущего состояния игры"""
        head_x = game.snake.body[0][0] // game.cell_size
        head_y = game.snake.body[0][1] // game.cell_size
        
        food_x = game.food.position[0] // game.cell_size
        food_y = game.food.position[1] // game.cell_size
        
        dir_left = game.snake.direction == LEFT
        dir_right = game.snake.direction == RIGHT
        dir_up = game.snake.direction == UP
        dir_down = game.snake.direction == DOWN
        
        danger_straight = 0
        danger_right = 0
        danger_left = 0
        
        # Рассчет опасностей
        # ... (реализация проверки препятствий)
        
        return np.array([
            # Положение еды относительно головы
            food_x < head_x,  # еда слева
            food_x > head_x,  # еда справа
            food_y < head_y,  # еда сверху
            food_y > head_y,  # еда снизу
            
            # Направление движения
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Опасности
            danger_straight,
            danger_right,
            danger_left
        ], dtype=int)
    
    def get_action(self, state):
        """Выбор действия с использованием ε-жадной стратегии"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(4)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Обучение на основе накопленного опыта"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = self.loss_fn(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class MLGame(SnakeGame):
    def __init__(self, field_size, speed):
        super().__init__(field_size, speed)
        self.agent = RLAgent(input_size=11, output_size=4)
        self.episode = 0
        self.total_reward = 0
    
    def update(self):
        state = self.agent.get_state(self)
        action = self.agent.get_action(state)
        
        # Преобразование действия в направление
        self.convert_action_to_direction(action)
        
        # Сохранение предыдущего состояния
        prev_head = self.snake.body[0]
        super().update()
        next_state = self.agent.get_state(self)
        
        # Расчет награды
        reward = self.calculate_reward(prev_head)
        self.total_reward += reward
        
        # Сохранение опыта
        self.agent.remember(state, action, reward, next_state, self.game_over)
        
        # Обучение
        self.agent.replay()
        
        if self.game_over:
            self.episode += 1
            print(f"Episode: {self.episode}, Total Reward: {self.total_reward}, Epsilon: {self.agent.epsilon:.2f}")
            self.total_reward = 0
            self.agent.update_target_model()
    
    def convert_action_to_direction(self, action):
        # Действия: [0 - прямо, 1 - направо, 2 - налево]
        directions = [UP, RIGHT, DOWN, LEFT]
        current_dir = self.snake.direction
        idx = directions.index(current_dir)
        
        if action == 0:  # Продолжать движение
            new_dir = current_dir
        elif action == 1:  # Поворот направо
            new_dir = directions[(idx + 1) % 4]
        elif action == 2:  # Поворот налево
            new_dir = directions[(idx - 1) % 4]
        elif action == 3:  # Противоположное направление
            new_dir = directions[(idx + 2) % 4]
        
        if self.is_direction_safe(new_dir):
            self.snake.direction = new_dir
    
    def calculate_reward(self, prev_head):
        # Награда за выживание
        reward = 0.1
        
        # Награда за съедение еды
        if self.snake.body[0] == self.food.position:
            reward += 10
            
        # Штраф за смерть
        if self.game_over:
            reward -= 10
            
        # Награда за приближение к еде
        prev_dist = self.get_distance(prev_head, self.food.position)
        new_dist = self.get_distance(self.snake.body[0], self.food.position)
        if new_dist < prev_dist:
            reward += 0.5
        else:
            reward -= 0.5
            
        return reward
    
    def get_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def is_direction_safe(self, direction):
        # Проверка безопасности направления
        head_x, head_y = self.snake.body[0]
        dx, dy = direction
        next_pos = (
            (head_x + dx * self.cell_size) % self.game_width,
            (head_y + dy * self.cell_size) % self.game_height
        )
        return next_pos not in self.snake.body[1:]

if __name__ == "__main__":
    field_size = (30, 20)
    speed = 100  # Максимальная скорость для обучения
    game = MLGame(field_size, speed)
    game.run()