import pygame
import random
import numpy as np

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

    def draw(self, screen):
        for i, (x, y) in enumerate(self.body):
            color = DARK_GREEN if i == 0 else GREEN
            pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

class Apple:
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, snake_body):
        while True:
            position = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
            if position not in snake_body:
                return position

    def draw(self, screen):
        pygame.draw.rect(screen, RED, (self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

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