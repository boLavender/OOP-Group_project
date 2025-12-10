import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def print_success_rate(rewards_per_episode):
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def random_argmax(q_values):
    top_value = np.max(q_values)
    ties = np.flatnonzero(q_values == top_value)
    return np.random.choice(ties)

def get_potential(state):
    row = state // 8
    col = state % 8
    goal_row, goal_col = 7, 7
    dist = abs(goal_row - row) + abs(goal_col - col)
    max_dist = 14
    return (max_dist - dist) / max_dist

def run_pbrs(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.random.uniform(low=0, high=0.001, size=(env.observation_space.n, env.action_space.n))
    else:
        f = open('frozen_lake8x8_pbrs.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    # --- 參數設定 ---
    # [復原] 回到實驗 5 的穩定設定
    learning_rate_a = 0.05       
    discount_factor_g = 0.99     
    
    epsilon = 1
    epsilon_decay_rate = 0.0000025 
    min_exploration_rate = 0.001
    
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]
        terminated = False
        truncated = False
        
        current_potential = get_potential(state)

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = random_argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)
            next_potential = get_potential(new_state)

            if is_training:
                # [🔥 實驗 7：放大 PBRS 訊號]
                # 這次我們有低學習率(0.05)護體，可以試著把導航訊號加倍
                # 讓 Agent 更強烈地想要"往右下角衝"
                pbrs_scale = 2.2
                shaping = pbrs_scale * (discount_factor_g * next_potential - current_potential)
                
                # 維持 -0.8 處罰
                modified_reward = reward
                if terminated and reward == 0:
                    modified_reward = -0.8
                
                total_reward = modified_reward + shaping

                q[state,action] = q[state,action] + learning_rate_a * (
                    total_reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state
            current_potential = next_potential 
            
            epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

        
        
        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
        plt.plot(sum_rewards)
        plt.title('Frozen Lake 8x8 (Scale 2.0)')
        plt.savefig('frozen_lake8x8_pbrs.png')
        
        f = open("frozen_lake8x8_pbrs.pkl","wb")
        pickle.dump(q, f)
        f.close()
    
    if not is_training:
        print_success_rate(rewards_per_episode)

if __name__ == '__main__':
    print("🚀 實驗 7: Scale 2.0 + Stable LR 0.05 ...")
    run_pbrs(15000, is_training=True, render=False)

    print("\n📊 評估 (Run 1)...")
    run_pbrs(1000, is_training=False, render=False)
    
    print("\n📊 評估 (Run 2)...")
    run_pbrs(1000, is_training=False, render=False)
