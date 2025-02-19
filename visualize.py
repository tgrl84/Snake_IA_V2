import pygame
import torch
from snake_game import Snake, Apple, get_state, WIDTH, HEIGHT, UP, DOWN, LEFT, RIGHT, CELL_SIZE, SCREEN_SIZE, BLACK, FPS
from snake_ai import DQNAgent

# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()

def visualize_game(model_path="snake_ai_final.pth"):
    # Charger l'agent et le modèle entraîné
    agent = DQNAgent(state_size=10, action_size=4)
    agent.model.load_state_dict(torch.load(model_path))
    agent.epsilon = 0  # Pas d'exploration, seulement exploitation
    
    # Initialiser le jeu
    snake = Snake()
    apple = Apple(snake.body)
    state = get_state(snake, apple)
    done = False
    
    while not done:
        # Gestion des événements (pour pouvoir quitter)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # Obtenir l'action de l'IA
        action = agent.act(state)
        directions = [UP, DOWN, LEFT, RIGHT]
        snake.direction = directions[action]
        snake.move()
        
        # Mettre à jour l'état
        next_state = get_state(snake, apple)
        
        # Vérifier les collisions
        if snake.body[0] == apple.position:
            snake.grow()
            apple = Apple(snake.body)
        elif (snake.body[0][0] < 0 or snake.body[0][0] >= WIDTH or
              snake.body[0][1] < 0 or snake.body[0][1] >= HEIGHT or
              snake.body[0] in snake.body[1:]):
            done = True
        
        # Mettre à jour l'état
        state = next_state
        
        # Affichage
        screen.fill(BLACK)
        snake.draw(screen)
        apple.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)
    
    print(f"Game over! Final score: {len(snake.body)}")
    pygame.quit()

if __name__ == "__main__":
    visualize_game() 