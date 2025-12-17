
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer

STATE_SPACE_LOW = np.array([0.0, 0.0, 50.0, 0.0, 1.0], dtype=np.float32)
STATE_SPACE_HIGH = np.array([100.0, 100.0, 2048.0, 0.5, 20.0], dtype=np.float32)

class DummyEnv(gym.Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=STATE_SPACE_LOW,
            high=STATE_SPACE_HIGH,
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
    def step(self, action): pass
    def reset(self, seed=None, options=None): pass

def test_crash():
    env = DummyEnv()
    model = DQN("MlpPolicy", env, learning_rate=1e-4, buffer_size=1000, learning_starts=10)
    
    # Fill buffer
    state = np.zeros(5, dtype=np.float32)
    for i in range(15):
        action =  model.predict(state, deterministic=False)[0]
        next_state = state
        reward = 1.0
        model.replay_buffer.add(state, next_state, action, reward, done=False, infos=[{}])
    
    print("Buffer filled. calling train()...")
    model.train(gradient_steps=1)
    print("Train successful!")

if __name__ == "__main__":
    test_crash()
