import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tom_utility import *

def evaluate_tomnet(model, num_samples=5, alpha=0.01):
    env = Gridworld()
    agent = RandomAgent(alpha)
    agent.reset()
    past_data = []

    # 蒐集 Npast 筆資料
    for _ in range(num_samples):
        state = env.reset()
        obs = agent.observe(state)
        action = agent.act(obs)
        past_data.append((state, action))

    # 選擇查詢狀態（ToMnet 預測目標）
    query_state = env.reset()

    # 使用最後一筆過去行為當作 input（可升級為平均或 LSTM）
    last_state, last_action = past_data[-1]
    state_tensor = torch.tensor(last_state, dtype=torch.float32).unsqueeze(0)
    action_tensor = F.one_hot(torch.tensor(
        last_action), 5).float().unsqueeze(0)
    query_tensor = torch.tensor(query_state, dtype=torch.float32).unsqueeze(0)

    # ToMnet 預測
    with torch.no_grad():
        logits = model(state_tensor, action_tensor, query_tensor)
        probs = F.softmax(logits, dim=-1)
        predicted_action = torch.argmax(probs, dim=-1).item()

    # 顯示基本資訊
    print("🔍 ToMnet prediction:")
    print(f" - distributions of actions: {probs.squeeze().numpy().round(3)}")
    print(
        f" - predicted action index: {predicted_action}（←=0, →=1, ↑=2, ↓=3, ·=4）")

    # 加入可視化
    print("🗺️ Query Gridworld:")
    # visualize_gridworld_with_action(query_tensor.squeeze().numpy(), predicted_action, title="ToMnet visualization")

    visualize_gridworld_with_history(
        query_tensor.squeeze().numpy(),
        predicted_action,
        [s for s, _ in past_data],
        [a for _, a in past_data],
        title="ToMnet visualization"
    )

    return predicted_action, probs


# 評估 ToMnet 對 mixture species 的預測品質（用 KL divergence)
def evaluate_tomnet_on_mixture(model, alpha_low=0.01, alpha_high=3.0, ratio=0.5, npast=5, num_test_agents=200):
    kl_values = []
    env = Gridworld()

    for _ in range(num_test_agents):
        alpha = alpha_low if np.random.rand() < ratio else alpha_high
        agent = RandomAgent(alpha)
        pi_true = agent.pi

        past_states, past_actions = [], []
        for _ in range(npast):
            s = env.reset()
            obs = agent.observe(s)
            a = agent.act(obs)
            past_states.append(s)
            past_actions.append(np.eye(5)[a])

        query_state = env.reset()

        s_tensor = torch.tensor(np.stack(past_states), dtype=torch.float32).unsqueeze(0)
        a_tensor = torch.tensor(np.stack(past_actions), dtype=torch.float32).unsqueeze(0)
        q_tensor = torch.tensor(query_state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits = model(s_tensor, a_tensor, q_tensor)
            probs = F.softmax(logits, dim=-1).squeeze().numpy()

        kl = np.sum(pi_true * (np.log(pi_true + 1e-8) - np.log(probs + 1e-8)))
        kl_values.append(kl)

    return np.mean(kl_values)

# 視覺化兩種單一 species 與混合 species 模型在不同測試上的 KL 效果
def plot_figure3d_kl_bars(kl_001_on_001, kl_3_on_3, kl_mix_on_mix):
    labels = ['ToMnet@α=0.01', 'ToMnet@α=3.0', 'ToMnet@Mixture']
    values = [kl_001_on_001, kl_3_on_3, kl_mix_on_mix]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title("Figure 3d: KL divergence on mixture agents")
    plt.ylabel("KL Divergence (lower is better)")
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() +
                 0.01, f"{val:.3f}", ha='center', va='bottom')
    plt.ylim(0, max(values) * 1.2)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_figure3d_kl_lines(kl_001_on_001, kl_3_on_3, kl_mix_on_mix):
    # 假設的 Test α 值
    test_alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    # 模擬的 ToMnet 預測 KL 散度數值
    kl_trained_001 = kl_001_on_001
    kl_trained_3 = kl_3_on_3
    kl_trained_mix = kl_mix_on_mix

    plt.figure(figsize=(4, 4))

    # 畫圖
    plt.plot(test_alphas, kl_trained_001, '-o', color='black', label='Trained α=0.01')
    plt.plot(test_alphas, kl_trained_3, '-o', color='skyblue', label='Trained α=3')
    plt.plot(test_alphas, kl_trained_mix, '-s', color='firebrick', label='Trained α=0.01,3')

    # 標示
    plt.xscale('log')
    plt.xlabel('Test α')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence (5 obs.)')

    plt.legend(title='Trained α')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


def exp_3_1_grid():
    
    model = ToMnet()

    if not torch.load(r"trained_model\tomnet.pth"):
        # === 訓練階段 ===
        train_tomnet()

    # === 推論階段 ===
    model.load_state_dict(torch.load(r"trained_model\tomnet.pth"))
    model.eval()  # 不啟用 dropout/batchnorm

    # 推論使用
    num_samples = 10
    alpha = 3
    evaluate_tomnet(model, num_samples, alpha)  # 你前面整合的推論+可視化函數


def exp_3_1_belief():

    # alpha_mix_dataset = ToMMixtureDataset(num_episodes=1000, alpha_low=0.01, alpha_high=3.0, ratio=0.5, npast=5)
    # alpha_low_dataset = ToMDatasetFlexible(num_episodes=1000, alpha=0.01)
    # alpha_high_dataset = ToMDatasetFlexible(num_episodes=1000, alpha=3.0)

    # === 訓練階段 ===
    print("training ToM ...")
    # train_tomnet(alpha_mix_dataset, "tomnet_mix")
    print("finish training mix")
    # train_tomnet(alpha_low_dataset, "tomnet_001")
    print("finish training 0.01")
    # train_tomnet(alpha_high_dataset, "tomnet_3")
    print("finish training 3")

    # 假設你有三個模型：
    model_001 = ToMnet()
    model_001.load_state_dict(torch.load(r"trained_model/tomnet_001.pth"))  # 已在 alpha=0.01 上訓練
    model_001.eval()
    
    model_3 = ToMnet()
    model_3.load_state_dict(torch.load(r"trained_model/tomnet_3.pth"))    # 已在 alpha=3 上訓練
    model_3.eval()
    
    model_mix = ToMnet()
    model_mix.load_state_dict(torch.load(r"trained_model/tomnet_mix.pth"))  # 已在混合 agents 上訓練
    model_mix.eval()

    # 繪圖評估三者在混合 agents 上的表現
    # kl_001_on_mix = evaluate_tomnet_on_mixture(model_001)
    # kl_3_on_mix = evaluate_tomnet_on_mixture(model_3)
    # kl_mix_on_mix = evaluate_tomnet_on_mixture(model_mix)
    # plot_figure3d_kl_bars(kl_001_on_mix, kl_3_on_mix, kl_mix_on_mix)

    # 定義測試用的 alpha 值（x 軸）
    test_alphas = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

    # 分別在不同的測試 α 上評估三個模型
    kl_001_curve = [evaluate_tomnet_on_mixture(model_001, alpha_low=α, alpha_high=α) for α in test_alphas]
    kl_3_curve =   [evaluate_tomnet_on_mixture(model_3, alpha_low=α, alpha_high=α) for α in test_alphas]
    kl_mix_curve = [evaluate_tomnet_on_mixture(model_mix, alpha_low=α, alpha_high=α) for α in test_alphas]
    plot_figure3d_kl_lines(kl_001_curve, kl_3_curve, kl_mix_curve)


if __name__ == "__main__":

    exp_3_1_grid()

    # exp_3_1_belief()