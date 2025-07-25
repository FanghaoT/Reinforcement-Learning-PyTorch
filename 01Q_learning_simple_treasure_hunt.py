"""
Simple Treasure Hunt - Exact Implementation from Blog Post
"Introduction to Reinforcement Learning - The Treasure Hunt"

A minimal implementation that exactly follows the blog post description.
"""

import numpy as np

# Parameters from the blog post
N_STATES = 6
ACTIONS = ['left', 'right'] 
EPSILON = 0.9   # greedy policy (Note: This means 90% exploitation, 10% exploration)
ALPHA = 0.1     # learning rate  
GAMMA = 0.9     # discount factor
MAX_EPISODES = 100
FRESH_TIME = 0.3

def build_q_table(n_states, actions):
    """Initialize Q-table with zeros"""
    table = np.zeros((n_states, len(actions)))
    return table

def choose_action(state, q_table):
    """
    Choose action using ε-greedy policy
    Note: Blog post has EPSILON=0.9 meaning 90% exploitation
    """
    state_actions = q_table[state, :]
    if (np.random.uniform() > EPSILON) or (np.all(state_actions == 0)):
        # Act non-greedy (explore) or when all actions have zero value
        action = np.random.choice(len(ACTIONS))
    else:
        # Act greedy (exploit)
        action = np.argmax(state_actions)
    return action

def get_env_feedback(S, A):
    """
    Environment feedback function
    S: current state, A: action (0=left, 1=right)
    Returns: next_state, reward
    """
    if A == 1:    # move right
        if S == N_STATES - 2:   # reached treasure (state 4 -> state 5)
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:   # move left  
        R = 0
        if S == 0:
            S_ = S  # stay at leftmost position
        else:
            S_ = S - 1
    return S_, R

def update_env(S, episode, step_counter):
    """Update environment visualization"""
    env_list = ['   '] * (N_STATES-1) + [' T ']  # ----T (treasure)
    if S == 'terminal':
        interaction = f'Episode {episode+1}: total_steps = {step_counter}'
        print(f'\r{interaction}', end='')
        print('\r                                ', end='')
        print(f'\r{interaction}', end='')
    else:
        env_list[S] = ' o '  # agent position
        interaction = '|' + '|'.join(env_list) + '|'
        print(f'\r{interaction}', end='')

def rl():
    """Main Q-learning function"""
    # Initialize Q-table
    q_table = build_q_table(N_STATES, ACTIONS)
    
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0  # initial state
        is_terminated = False
        update_env(S, episode, step_counter)
        
        while not is_terminated:
            # Choose action
            A = choose_action(S, q_table)
            
            # Take action and get feedback
            S_, R = get_env_feedback(S, A)
            
            # Q-learning update rule
            q_predict = q_table[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * np.max(q_table[S_, :])
            else:
                q_target = R
                is_terminated = True
            
            # Update Q-table: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
            q_table[S, A] += ALPHA * (q_target - q_predict)
            
            # Move to next state
            S = S_
            
            # Update environment
            update_env(S, episode, step_counter+1)
            step_counter += 1
    
    return q_table

def print_q_table(q_table):
    """Print the final Q-table"""
    print(f'\n\nFinal Q-table:')
    print('State | Left  | Right')
    print('------|-------|-------')
    for i in range(N_STATES):
        print(f'  {i}   | {q_table[i,0]:.3f} | {q_table[i,1]:.3f}')

def print_policy(q_table):
    """Print the learned policy"""
    print(f'\nLearned Policy:')
    for i in range(N_STATES):
        best_action = ACTIONS[np.argmax(q_table[i, :])]
        print(f'State {i}: {best_action}')

if __name__ == "__main__":
    print("Simple Treasure Hunt Q-Learning")
    print("=" * 40)
    print("Environment: [ o |   |   |   |   | T ]")
    print("             0   1   2   3   4   5")
   

    # Run Q-learning
    final_q_table = rl()
    
    # Display results
    print_q_table(final_q_table)
    print_policy(final_q_table)