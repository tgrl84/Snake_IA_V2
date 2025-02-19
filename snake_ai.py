import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import pygame
import os

# Initialisation de Pygame
pygame.init()

# Constantes
WIDTH, HEIGHT = 10, 10
CELL_SIZE = 40
SCREEN_SIZE = (WIDTH * CELL_SIZE, HEIGHT * CELL_SIZE)
FPS = 10

# Couleurs
DARK_GREEN = (0, 100, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Initialisation de l'écran
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Snake AI")
clock = pygame.time.Clock()

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

class Snake:
    def __init__(self):
        self.body = [(5, 5), (4, 5), (3, 5)]
        self.direction = RIGHT

    def move(self):
        head = self.body[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.body.insert(0, new_head)
        self.body.pop()

    def grow(self):
        tail = self.body[-1]
        self.body.append(tail)

    def draw(self):
        for i, (x, y) in enumerate(self.body):
            color = DARK_GREEN if i == 0 else GREEN
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Apple:
    def __init__(self):
        self.position = self.random_position()

    def random_position(self):
        return (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))

    def draw(self):
        pygame.draw.rect(screen, RED, (self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = DQN(state_size, 128, action_size)
        self.target_model = DQN(state_size, 128, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.update_target_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.model.device)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch])).to(self.model.device)
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.model.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.model.device)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch])).to(self.model.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.model.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def get_state(snake, apple):
    head = snake.body[0]
    return np.array([
        head[0], head[1],
        apple.position[0] - head[0],
        apple.position[1] - head[1],
        snake.direction[0], snake.direction[1],
        *[1 if (head[0] + dx < 0 or head[0] + dx >= WIDTH or 
                 head[1] + dy < 0 or head[1] + dy >= HEIGHT or 
                 (head[0] + dx, head[1] + dy) in snake.body) 
          else 0 for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]]
    ], dtype=np.float32)

def train_ai(episodes=1000, render_every=100, save_every=100):
    agent = DQNAgent(state_size=10, action_size=4)
    scores = []
    avg_scores = []
    
    for episode in range(episodes):
        snake = Snake()
        apple = Apple()
        state = get_state(snake, apple)
        total_reward = 0
        done = False
        
        while not done:
            if episode % render_every == 0:
                pygame.event.pump()
                screen.fill(BLACK)
                snake.draw()
                apple.draw()
                pygame.display.flip()
                clock.tick(FPS)
            
            action = agent.act(state)
            directions = [UP, DOWN, LEFT, RIGHT]
            snake.direction = directions[action]
            snake.move()
            
            next_state = get_state(snake, apple)
            reward = 0
            
            # Reward system
            if snake.body[0] == apple.position:
                reward = 10
                snake.grow()
                apple.position = apple.random_position()
            elif (snake.body[0][0] < 0 or snake.body[0][0] >= WIDTH or
                  snake.body[0][1] < 0 or snake.body[0][1] >= HEIGHT or
                  snake.body[0] in snake.body[1:]):
                reward = -10
                done = True
            else:
                # Encourage moving towards apple
                dist_before = abs(state[2]) + abs(state[3])
                dist_after = abs(next_state[2]) + abs(next_state[3])
                reward = 1 if dist_after < dist_before else -1
            
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
        
        scores.append(len(snake.body))
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % save_every == 0:
            torch.save(agent.model.state_dict(), f"snake_ai_{episode}.pth")
            agent.update_target_model()
        
        print(f"Episode: {episode+1}/{episodes}, Score: {len(snake.body)}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Plot results
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Avg Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_results.png')
    plt.show()

if __name__ == "__main__":
    train_ai()