"""
Treasure Hunt Q-Learning Implementation
Based on the blog post: "Introduction to Reinforcement Learning - The Treasure Hunt"

This implements a simple 1D world where an agent learns to find treasure using Q-learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class TreasureHuntEnvironment:
    """
    A simple 1D environment where:
    - Agent starts at position 0
    - Treasure is at position 5
    - Agent can move left or right
    - Agent gets +1 reward when reaching treasure
    """
    
    def __init__(self, world_size=6):
        self.world_size = world_size
        self.treasure_position = world_size - 1  # Treasure at rightmost position
        self.agent_position = 0
        
    def reset(self):
        """Reset agent to starting position"""
        self.agent_position = 0
        return self.agent_position
    
    def step(self, action):
        """
        Take an action and return (next_state, reward, done)
        Actions: 0 = left, 1 = right
        """
        if action == 0:  # Move left
            self.agent_position = max(0, self.agent_position - 1)
        elif action == 1:  # Move right
            self.agent_position = min(self.world_size - 1, self.agent_position + 1)
        
        # Check if reached treasure
        if self.agent_position == self.treasure_position:
            reward = 1.0
            done = True
        else:
            reward = 0.0
            done = False
            
        return self.agent_position, reward, done
    
    def render(self):
        """Visualize the current state"""
        world = ['   '] * self.world_size
        world[self.agent_position] = ' o '
        world[self.treasure_position] = ' T ' if self.agent_position != self.treasure_position else 'oT '
        
        print('|' + '|'.join(world) + '|')
        print(' ' + '   '.join([str(i) for i in range(self.world_size)]))

class QLearningAgent:
    """
    Q-Learning agent implementation following the algorithm from the blog post
    """
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        Initialize Q-learning agent
        
        Note: The blog post mentions EPSILON=0.9, but this would mean 90% exploration
        which is unusual. Typically epsilon=0.1 (10% exploration) is more common.
        I'll use 0.1 here, but you can change it to 0.9 to match the blog exactly.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = learning_rate      # Learning rate (α)
        self.gamma = discount_factor    # Discount factor (γ)
        self.epsilon = epsilon          # Exploration rate (ε)
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track learning progress
        self.episode_rewards = []
        self.episode_steps = []
    
    def choose_action(self, state):
        """
        Choose action using ε-greedy policy
        - With probability (1-ε): Choose best action (exploit)
        - With probability ε: Choose random action (explore)
        """
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            action = np.random.randint(self.n_actions)
        else:
            # Exploit: choose best action
            action = np.argmax(self.q_table[state])
        
        return action
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using the Q-learning update rule:
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table[state][action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[next_state])
        
        # Calculate temporal difference target
        td_target = reward + self.gamma * max_next_q
        
        # Calculate temporal difference error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state][action] = current_q + self.alpha * td_error
    
    def train(self, env, num_episodes=500, max_steps_per_episode=100, verbose=True):
        """Train the agent using Q-learning"""
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(max_steps_per_episode):
                # Choose action using ε-greedy policy
                action = self.choose_action(state)
                
                # Take action and observe result
                next_state, reward, done = env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                # Update tracking variables
                total_reward += reward
                steps += 1
                state = next_state
                
                # End episode if treasure found
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            
            # Print progress
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_steps = np.mean(self.episode_steps[-50:])
                print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, Avg Steps = {avg_steps:.1f}")
    
    def display_q_table(self):
        """Display the learned Q-table"""
        df = pd.DataFrame(self.q_table, 
                         columns=['Left', 'Right'],
                         index=[f'State {i}' for i in range(self.n_states)])
        print("\nLearned Q-Table:")
        print(df.round(3))
    
    def display_policy(self):
        """Display the learned policy"""
        print("\nLearned Policy:")
        for state in range(self.n_states):
            best_action = np.argmax(self.q_table[state])
            action_name = "Left" if best_action == 0 else "Right"
            print(f"State {state}: {action_name}")

def demonstrate_learning(agent, env, num_demos=5):
    """Demonstrate the learned policy"""
    print(f"\n=== Demonstrating Learned Policy ===")
    
    for demo in range(num_demos):
        print(f"\nDemo {demo + 1}:")
        state = env.reset()
        env.render()
        
        total_reward = 0
        steps = 0
        
        for step in range(20):  # Max 20 steps per demo
            # Choose best action (no exploration)
            action = np.argmax(agent.q_table[state])
            action_name = "left" if action == 0 else "right"
            
            print(f"Step {step + 1}: Action = {action_name}")
            
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.5)  # Pause for visualization
            
            if done:
                print(f"🎉 Treasure found in {steps} steps! Total reward: {total_reward}")
                break
                
            state = next_state
        
        if not done:
            print("❌ Failed to find treasure within step limit")

def plot_learning_progress(agent):
    """Plot the learning progress"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot episode rewards
    ax1.plot(agent.episode_rewards)
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode steps (smoothed)
    window_size = 20
    if len(agent.episode_steps) >= window_size:
        smoothed_steps = np.convolve(agent.episode_steps, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        ax2.plot(range(window_size-1, len(agent.episode_steps)), smoothed_steps)
        ax2.set_title(f'Steps to Find Treasure (Smoothed, window={window_size})')
    else:
        ax2.plot(agent.episode_steps)
        ax2.set_title('Steps to Find Treasure')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the treasure hunt Q-learning example"""
    
    print("🏴‍☠️ Treasure Hunt Q-Learning Example 🏴‍☠️")
    print("=" * 50)
    
    # Create environment and agent
    env = TreasureHuntEnvironment(world_size=6)
    agent = QLearningAgent(n_states=6, n_actions=2, 
                          learning_rate=0.1, 
                          discount_factor=0.9, 
                          epsilon=0.1)  # 10% exploration
    
    print("Initial environment:")
    env.render()
    print("\nStarting Q-learning training...")
    
    # Train the agent
    agent.train(env, num_episodes=500, verbose=True)
    
    # Display results
    agent.display_q_table()
    agent.display_policy()
    
    # Demonstrate the learned policy
    demonstrate_learning(agent, env, num_demos=3)
    
    # Plot learning progress
    plot_learning_progress(agent)
    
    print("\n🎯 Training completed! The agent has learned to find the treasure efficiently.")

if __name__ == "__main__":
    main() 