import numpy as np
import json
import os
from compress import N_ACTIONS, compute_reward

# ── Q-table agent ──
# State = signal type (0=ECG, 1=Accelerometer, 2=Temperature)
# Action = which compression params to use (0 to 23)
# Q[state][action] = expected reward for taking that action in that state

N_STATES = 3  # three signal types


class QLearningAgent:
    def __init__(
        self,
        lr=0.1,          # learning rate — how fast to update Q values
        gamma=0.9,       # discount factor — how much future rewards matter
        epsilon=1.0,     # exploration rate — starts at 1.0 (fully random)
        epsilon_min=0.05,
        epsilon_decay=0.995,
        q_table_path="q_table.json",
    ):
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table_path  = q_table_path

        # Try to load an existing Q-table, otherwise start fresh with zeros
        if os.path.exists(q_table_path):
            self.q_table = np.array(json.load(open(q_table_path)))
            print(f"Loaded Q-table from {q_table_path}")
        else:
            self.q_table = np.zeros((N_STATES, N_ACTIONS))
            print("Started fresh Q-table")

        # Track history for the dashboard
        self.episode       = 0
        self.reward_history = []
        self.action_history = []

    def choose_action(self, state):
        """
        Epsilon-greedy action selection:
          - With probability epsilon: pick a RANDOM action (explore)
          - Otherwise: pick the action with highest Q value (exploit)

        Early in training epsilon is high → lots of exploration.
        Over time it decays → more exploitation of learned knowledge.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)  # random
        return int(np.argmax(self.q_table[state]))  # best known

    def update(self, state, action, reward, next_state):
        """
        Q-learning update rule:
          Q[s][a] = Q[s][a] + lr * (reward + gamma * max(Q[s']) - Q[s][a])

        In plain English:
          "Update our estimate of how good action a was in state s,
           using the actual reward we got plus our best guess of future rewards."
        """
        best_next = np.max(self.q_table[next_state])
        current_q = self.q_table[state][action]
        self.q_table[state][action] = current_q + self.lr * (
            reward + self.gamma * best_next - current_q
        )

        # Decay epsilon after each update
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Log for dashboard
        self.reward_history.append(round(reward, 4))
        self.action_history.append(action)
        self.episode += 1

    def save(self):
        """Saves Q-table to disk so training persists across runs."""
        json.dump(self.q_table.tolist(), open(self.q_table_path, "w"))

    def get_stats(self):
        """Returns a snapshot of agent state — sent to the dashboard."""
        recent_rewards = self.reward_history[-50:] if self.reward_history else []
        return {
            "episode":        self.episode,
            "epsilon":        round(self.epsilon, 4),
            "avg_reward":     round(float(np.mean(recent_rewards)), 4) if recent_rewards else 0,
            "reward_history": recent_rewards,
            "q_table":        self.q_table.tolist(),
        }

    def best_action_for(self, state):
        """Returns the currently best-known action for a given state."""
        return int(np.argmax(self.q_table[state]))


# ── Quick test ──
if __name__ == "__main__":
    agent = QLearningAgent()

    # Simulate 200 training steps
    print("Running 200 training steps...")
    for i in range(200):
        state  = np.random.randint(N_STATES)
        action = agent.choose_action(state)

        # Fake reward — in real usage this comes from compress.compute_reward()
        fake_cr  = np.random.uniform(1.5, 4.0)
        fake_err = np.random.uniform(0.0001, 0.01)
        reward   = compute_reward(fake_cr, fake_err)

        next_state = np.random.randint(N_STATES)
        agent.update(state, action, reward, next_state)

    stats = agent.get_stats()
    print(f"Episodes:   {stats['episode']}")
    print(f"Epsilon:    {stats['epsilon']}")
    print(f"Avg reward: {stats['avg_reward']}")
    print("agent.py OK")