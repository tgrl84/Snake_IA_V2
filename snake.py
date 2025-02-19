import pygame
import random

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
pygame.display.set_caption("Snake")
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
    def __init__(self, snake_body):
        self.position = self.random_position(snake_body)

    def random_position(self, snake_body):
        while True:
            position = (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
            if position not in snake_body:
                return position

    def draw(self):
        pygame.draw.rect(screen, RED, (self.position[0] * CELL_SIZE, self.position[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

def main():
    snake = Snake()
    apple = Apple(snake.body)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and snake.direction != DOWN:
                    snake.direction = UP
                elif event.key == pygame.K_DOWN and snake.direction != UP:
                    snake.direction = DOWN
                elif event.key == pygame.K_LEFT and snake.direction != RIGHT:
                    snake.direction = LEFT
                elif event.key == pygame.K_RIGHT and snake.direction != LEFT:
                    snake.direction = RIGHT

        snake.move()

        # Collision avec la pomme
        if snake.body[0] == apple.position:
            snake.grow()
            apple.position = apple.random_position(snake.body)

        # Collision avec les bords ou soi-même
        head = snake.body[0]
        if (head[0] < 0 or head[0] >= WIDTH or
            head[1] < 0 or head[1] >= HEIGHT or
            head in snake.body[1:]):
            pygame.quit()
            return

        # Affichage
        screen.fill(BLACK)
        snake.draw()
        apple.draw()
        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()