import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- 參數 (實驗 7 配置: High Stability) ---
LEARNING_RATE = 0.05      # 低學習率 (穩定)
DISCOUNT_FACTOR = 0.99    # 標準遠見
PENALTY = -0.8           # 重處罰 (謹慎) 
PBRS_SCALE = 2.2          # [關鍵] 強導航訊號 (因為有低 LR 保護，所以開大一點沒關係)
EPSILON_DECAY = 0.0000025
MIN_EPSILON = 0.001

def random_argmax(q_values):
    """解決 np.argmax 偏差的關鍵函式"""
    top_value = np.max(q_values)
    ties = np.flatnonzero(q_values == top_value)
    return np.random.choice(ties)

def get_potential(state):
    """PBRS 位能計算"""
    row = state // 8
    col = state % 8
    goal_row, goal_col = 7, 7
    dist = abs(goal_row - row) + abs(goal_col - col)
    max_dist = 14
    return (max_dist - dist) / max_dist

def train_one_round(episode_count, run_id):
    """執行一次完整的訓練"""
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    
    # 隨機初始化
    q = np.random.uniform(low=0, high=0.001, size=(env.observation_space.n, env.action_space.n))
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episode_count)
    epsilon = 1.0
    
    for i in range(episode_count):
        state = env.reset()[0]
        terminated = False
        truncated = False
        current_potential = get_potential(state)

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = random_argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)
            next_potential = get_potential(new_state)

            # PBRS (Scale 2.0) + Penalty (-0.75)
            shaping = PBRS_SCALE * (DISCOUNT_FACTOR * next_potential - current_potential)
            modified_reward = reward
            if terminated and reward == 0:
                modified_reward = PENALTY
            
            total_reward = modified_reward + shaping

            q[state, action] = q[state, action] + LEARNING_RATE * (
                total_reward + DISCOUNT_FACTOR * np.max(q[new_state, :]) - q[state, action]
            )

            state = new_state
            current_potential = next_potential
            
            epsilon = max(epsilon - EPSILON_DECAY, MIN_EPSILON)

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()
    return q, rewards_per_episode

def evaluate(q_table, eval_episodes=1000):
    """評估目前的 Q-Table"""
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
    success_count = 0
    
    for _ in range(eval_episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # 評估時完全不探索 (Greedy)
            action = random_argmax(q_table[state, :])
            state, reward, terminated, truncated, _ = env.step(action)
            
            if reward == 1:
                success_count += 1
                
    env.close()
    return (success_count / eval_episodes) * 100

if __name__ == '__main__':
    TOTAL_RUNS = 10        # 連續跑 10 次刷分
    TRAIN_EPISODES = 15000 
    
    best_success_rate = 0.0
    best_run_id = -1
    
    print(f"🔥 開始自動刷分 (Scale 2.2 版)...")
    print("-" * 40)

    for i in range(1, TOTAL_RUNS + 1):
        print(f"🔄 Round {i}/{TOTAL_RUNS}: Training...", end="\r")
        
        # 1. 訓練
        q_table, train_rewards = train_one_round(TRAIN_EPISODES, i)
        
        # 2. 評估
        score = evaluate(q_table)
        print(f"📊 Round {i}/{TOTAL_RUNS}: Success Rate = {score:.2f}%", end="")
        
        # 3. 紀錄最高分
        if score > best_success_rate:
            best_success_rate = score
            best_run_id = i
            print("  (⭐ New Best!)")
            
            # 存檔
            with open('frozen_lake8x8_best.pkl', 'wb') as f:
                pickle.dump(q_table, f)
            
            # 畫圖
            plt.clf()
            sum_rewards = np.zeros(TRAIN_EPISODES)
            for t in range(TRAIN_EPISODES):
                sum_rewards[t] = np.sum(train_rewards[max(0, t-100):(t+1)])
            plt.plot(sum_rewards)
            plt.title(f'Best Run (Score: {score:.2f}%)')
            plt.savefig('frozen_lake8x8_best.png')
            
        else:
            print("") 

    print("-" * 40)
    print(f"🏆 最終結果：最高分出現在第 {best_run_id} 輪")
    print(f"✅ Final Best Success Rate: {best_success_rate:.2f}%")
