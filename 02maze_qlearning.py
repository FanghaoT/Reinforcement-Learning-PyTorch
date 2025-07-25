"""
Maze Navigation Q-Learning Implementation
Based on the blog post: "Tabular Q-Learning: Navigating the Maze"

This implements a grid world maze where an agent learns to navigate obstacles
and find the optimal path to the goal using Q-learning with dynamic Q-table construction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os

class MazeEnvironment:
    """
    Grid world maze environment where:
    - Agent (red rectangle) starts at a designated position
    - Goal (yellow oval) is at the target location
    - Black squares are walls/obstacles
    - White squares are valid paths
    """
    
    def __init__(self, maze_size=(6, 6)):
        self.maze_size = maze_size
        self.rows, self.cols = maze_size
        
        # Define maze layout (0 = path, 1 = wall)
        self.maze = np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0, 0]
        ])
        
        self.start_pos = (0, 0)  # Agent starting position
        self.goal_pos = (5, 5)   # Goal position
        self.agent_pos = self.start_pos
        
        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def reset(self):
        """Reset agent to starting position"""
        self.agent_pos = self.start_pos
        return self._get_state_key(self.agent_pos)
    
    def _get_state_key(self, pos):
        """Convert position to state key for Q-table"""
        return f"({pos[0]},{pos[1]})"
    
    def _is_valid_position(self, pos):
        """Check if position is within bounds and not a wall"""
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.maze[row, col] == 0
    
    def step(self, action):
        """
        Take an action and return (next_state, reward, done, info)
        """
        current_pos = self.agent_pos
        row_change, col_change = self.action_effects[action]
        new_pos = (current_pos[0] + row_change, current_pos[1] + col_change)
        
        # Check if new position is valid
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            # Hit wall or boundary - stay in current position
            new_pos = current_pos
            self.agent_pos = current_pos
        
        # Calculate reward
        if self.agent_pos == self.goal_pos:
            reward = 1.0  # Reached goal
            done = True
        elif new_pos == current_pos and action is not None:
            reward = -0.1  # Hit wall penalty
            done = False
        else:
            reward = -0.01  # Small step penalty to encourage efficiency
            done = False
        
        next_state = self._get_state_key(self.agent_pos)
        return next_state, reward, done, {'hit_wall': new_pos == current_pos}
    
    def render(self, title="Maze"):
        """Visualize the maze with agent and goal"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Draw maze
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i, j] == 1:  # Wall
                    rect = patches.Rectangle((j, self.rows-1-i), 1, 1, 
                                           linewidth=1, edgecolor='black', 
                                           facecolor='black')
                    ax.add_patch(rect)
                else:  # Path
                    rect = patches.Rectangle((j, self.rows-1-i), 1, 1, 
                                           linewidth=1, edgecolor='gray', 
                                           facecolor='white')
                    ax.add_patch(rect)
        
        # Draw goal (yellow oval)
        goal_row, goal_col = self.goal_pos
        goal_circle = patches.Ellipse((goal_col + 0.5, self.rows-1-goal_row + 0.5), 
                                    0.6, 0.6, facecolor='yellow', edgecolor='orange')
        ax.add_patch(goal_circle)
        
        # Draw agent (red rectangle)
        agent_row, agent_col = self.agent_pos
        agent_rect = patches.Rectangle((agent_col + 0.2, self.rows-1-agent_row + 0.2), 
                                     0.6, 0.6, facecolor='red', edgecolor='darkred')
        ax.add_patch(agent_rect)
        
        ax.set_xlim(0, self.cols)
        ax.set_ylim(0, self.rows)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            patches.Patch(color='red', label='Agent'),
            patches.Patch(color='yellow', label='Goal'),
            patches.Patch(color='black', label='Wall'),
            patches.Patch(color='white', label='Path', edgecolor='gray')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        return fig, ax

class MazeQLearningAgent:
    """
    Q-Learning agent with dynamic Q-table construction for maze navigation
    """
    
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # List of action names
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        
        # Dynamic Q-table using pandas DataFrame
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        # Learning statistics
        self.episode_rewards = []
        self.episode_steps = []
        self.wall_hits = []
    
    def check_state_exist(self, state):
        """Add new state to Q-table if it doesn't exist"""
        if state not in self.q_table.index:
            # Append new state to q table
            new_state = pd.Series([0]*len(self.actions), 
                                index=self.q_table.columns, 
                                name=state)
            self.q_table = pd.concat([self.q_table, new_state.to_frame().T])
    
    def choose_action(self, observation):
        """
        Choose action using ε-greedy policy
        """
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # Exploit: choose best known action
            state_action = self.q_table.loc[observation, :]
            # Handle ties by random selection among best actions
            max_actions = state_action[state_action == np.max(state_action)]
            action = np.random.choice(max_actions.index)
        else:
            # Explore: try random action
            action = np.random.choice(self.actions)
        
        return self.actions.index(action)  # Return action index
    
    def learn(self, s, a, r, s_):
        """
        Q-learning update rule with dynamic state handling
        """
        self.check_state_exist(s_)
        
        q_predict = self.q_table.loc[s, self.actions[a]]
        
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
            
        # Update Q-value
        self.q_table.loc[s, self.actions[a]] += self.lr * (q_target - q_predict)
    
    def train(self, env, num_episodes=100, max_steps_per_episode=200, verbose=True):
        """Train the agent in the maze environment"""
        
        print("🎯 Starting Q-Learning Training in Maze")
        print("=" * 50)
        
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            wall_hits_episode = 0
            
            for step in range(max_steps_per_episode):
                # Choose action
                action = self.choose_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Count wall hits
                if info.get('hit_wall', False):
                    wall_hits_episode += 1
                
                # Learn from experience
                self.learn(state, action, reward, next_state)
                
                # Update tracking
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.wall_hits.append(wall_hits_episode)
            
            # Print progress
            if verbose and (episode + 1) % 20 == 0:
                avg_reward = np.mean(self.episode_rewards[-20:])
                avg_steps = np.mean(self.episode_steps[-20:])
                avg_walls = np.mean(self.wall_hits[-20:])
                success_rate = sum([1 for r in self.episode_rewards[-20:] if r > 0]) / 20 * 100
                
                print(f"Episode {episode + 1:3d}: "
                      f"Avg Reward = {avg_reward:6.3f}, "
                      f"Avg Steps = {avg_steps:5.1f}, "
                      f"Avg Wall Hits = {avg_walls:4.1f}, "
                      f"Success Rate = {success_rate:5.1f}%")
    
    def display_q_table(self, max_states=20):
        """Display the learned Q-table (limited for readability)"""
        print(f"\nLearned Q-Table (showing first {max_states} states):")
        print("=" * 80)
        display_table = self.q_table.head(max_states).round(3)
        print(display_table.to_string())
        print(f"\nTotal states discovered: {len(self.q_table)}")
    
    def display_policy(self, max_states=20):
        """Display the learned policy"""
        print(f"\nLearned Policy (showing first {max_states} states):")
        print("=" * 40)
        
        states_to_show = min(max_states, len(self.q_table))
        for i, state in enumerate(self.q_table.index[:states_to_show]):
            best_action_idx = self.q_table.loc[state, :].idxmax()
            print(f"State {state}: {best_action_idx}")

def demonstrate_learning_phases(agent, env, episodes_per_phase=20):
    """Demonstrate the three learning phases mentioned in the blog post"""
    
    phases = [
        (0, episodes_per_phase, "Phase 1: Chaotic Exploration"),
        (episodes_per_phase, episodes_per_phase*2, "Phase 2: Path Discovery"), 
        (episodes_per_phase*2, episodes_per_phase*3, "Phase 3: Route Optimization")
    ]
    
    print("\n" + "="*60)
    print("DEMONSTRATING LEARNING PHASES")
    print("="*60)
    
    for start_ep, end_ep, phase_name in phases:
        print(f"\n{phase_name} (Episodes {start_ep+1}-{end_ep})")
        print("-" * 50)
        
        if start_ep < len(agent.episode_rewards):
            phase_rewards = agent.episode_rewards[start_ep:min(end_ep, len(agent.episode_rewards))]
            phase_steps = agent.episode_steps[start_ep:min(end_ep, len(agent.episode_steps))]
            phase_walls = agent.wall_hits[start_ep:min(end_ep, len(agent.wall_hits))]
            
            if phase_rewards:
                print(f"Average Reward: {np.mean(phase_rewards):.3f}")
                print(f"Average Steps: {np.mean(phase_steps):.1f}")
                print(f"Average Wall Hits: {np.mean(phase_walls):.1f}")
                success_rate = sum([1 for r in phase_rewards if r > 0]) / len(phase_rewards) * 100
                print(f"Success Rate: {success_rate:.1f}%")

def plot_learning_progress(agent):
    """Plot the learning progress showing the three phases"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episodes = range(1, len(agent.episode_rewards) + 1)
    
    # Plot 1: Episode Rewards
    ax1.plot(episodes, agent.episode_rewards, alpha=0.6)
    # Add moving average
    if len(agent.episode_rewards) >= 10:
        moving_avg = np.convolve(agent.episode_rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(10, len(agent.episode_rewards) + 1), moving_avg, 'r-', linewidth=2, label='Moving Average (10)')
        ax1.legend()
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Steps per Episode
    ax2.plot(episodes, agent.episode_steps, alpha=0.6, color='green')
    if len(agent.episode_steps) >= 10:
        moving_avg_steps = np.convolve(agent.episode_steps, np.ones(10)/10, mode='valid')
        ax2.plot(range(10, len(agent.episode_steps) + 1), moving_avg_steps, 'g-', linewidth=2, label='Moving Average (10)')
        ax2.legend()
    ax2.set_title('Steps to Complete Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Wall Hits per Episode
    ax3.plot(episodes, agent.wall_hits, alpha=0.6, color='orange')
    if len(agent.wall_hits) >= 10:
        moving_avg_walls = np.convolve(agent.wall_hits, np.ones(10)/10, mode='valid')
        ax3.plot(range(10, len(agent.wall_hits) + 1), moving_avg_walls, 'orange', linewidth=2, label='Moving Average (10)')
        ax3.legend()
    ax3.set_title('Wall Hits per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Wall Hits')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success Rate (rolling window)
    window_size = 20
    if len(agent.episode_rewards) >= window_size:
        success_rates = []
        for i in range(window_size, len(agent.episode_rewards) + 1):
            window_rewards = agent.episode_rewards[i-window_size:i]
            success_rate = sum([1 for r in window_rewards if r > 0]) / len(window_rewards) * 100
            success_rates.append(success_rate)
        
        ax4.plot(range(window_size, len(agent.episode_rewards) + 1), success_rates, 'purple', linewidth=2)
        ax4.set_title(f'Success Rate (Rolling {window_size}-Episode Window)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_learned_policy(agent, env, num_demos=3):
    """Demonstrate the learned policy"""
    print(f"\n{'='*60}")
    print("DEMONSTRATING LEARNED POLICY")
    print('='*60)
    
    for demo in range(num_demos):
        print(f"\n--- Demo {demo + 1} ---")
        state = env.reset()
        
        # Show initial maze
        if demo == 0:
            fig, ax = env.render(f"Demo {demo + 1}: Initial Position")
            plt.show()
        
        total_reward = 0
        steps = 0
        path = [env.agent_pos]
        
        for step in range(50):  # Max 50 steps per demo
            # Choose best action (no exploration)
            agent.check_state_exist(state)
            action_idx = agent.q_table.loc[state, :].idxmax()
            action = agent.actions.index(action_idx)
            
            next_state, reward, done, info = env.step(action)
            path.append(env.agent_pos)
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                print(f"🎉 Goal reached in {steps} steps! Total reward: {total_reward:.3f}")
                break
        
        if not done:
            print(f"Failed to reach goal within step limit. Steps: {steps}, Reward: {total_reward:.3f}")
        
        # Show final position
        if demo == 0:
            fig, ax = env.render(f"Demo {demo + 1}: Final Position")
            # Draw path
            if len(path) > 1:
                path_x = [pos[1] + 0.5 for pos in path]
                path_y = [env.rows - 1 - pos[0] + 0.5 for pos in path]
                ax.plot(path_x, path_y, 'b--', linewidth=2, alpha=0.7, label='Agent Path')
                ax.legend()
            plt.show()

def main():
    """Main function to run the maze Q-learning example"""
    
    print("Maze Navigation Q-Learning Example")
    print("=" * 60)
    
    # Create environment and agent
    env = MazeEnvironment()
    agent = MazeQLearningAgent(
        actions=['up', 'down', 'left', 'right'],
        learning_rate=0.01,
        reward_decay=0.9,
        e_greedy=0.9
    )
    
    # Show initial maze
    print("Initial Maze Layout:")
    fig, ax = env.render("Initial Maze")
    plt.show()
    
    # Train the agent
    agent.train(env, num_episodes=300, verbose=True)
    
    # Analyze learning phases
    demonstrate_learning_phases(agent, env)
    
    # Display results
    agent.display_q_table(max_states=15)
    agent.display_policy(max_states=15)
    
    # Demonstrate learned policy
    demonstrate_learned_policy(agent, env, num_demos=2)
    
    # Plot learning progress
    plot_learning_progress(agent)
    
    print(f"Training completed! Agent learned to navigate the maze.")
    print(f"Total states discovered: {len(agent.q_table)}")
    print(f"Q-table size: {agent.q_table.shape}")

if __name__ == "__main__":
    main() 