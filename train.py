import numpy as np
import matplotlib.pyplot as plt
import pygame
import torch
from snake_game import Snake, Apple, get_state, WIDTH, HEIGHT, UP, DOWN, LEFT, RIGHT, CELL_SIZE, SCREEN_SIZE, BLACK, FPS
from snake_ai import DQNAgent

# Initialisation de Pygame
pygame.init()
screen = pygame.display.set_mode(SCREEN_SIZE)
clock = pygame.time.Clock()

def train_ai(episodes=2000, render_every=100, save_every=500):
    agent = DQNAgent(state_size=10, action_size=4)
    scores = []
    avg_scores = []
    
    # Utiliser CUDA si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    for episode in range(episodes):
        snake = Snake()
        apple = Apple(snake.body)
        state = get_state(snake, apple)
        total_reward = 0
        done = False
        
        while not done:
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
                apple.position = apple.random_position(snake.body)
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
            
            # Réduire la fréquence des replays
            if episode % 2 == 0:
                agent.replay()
        
        scores.append(len(snake.body))
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)
        
        if episode % save_every == 0:
            torch.save(agent.model.state_dict(), f"snake_ai_{episode}.pth")
            agent.update_target_model()
            print(f"Episode: {episode+1}/{episodes}, Score: {len(snake.body)}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}")
    
    # Sauvegarder le modèle final
    final_model_path = "snake_ai_final.pth"
    torch.save(agent.model.state_dict(), final_model_path)
    print(f"Modèle final sauvegardé dans {final_model_path}")

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