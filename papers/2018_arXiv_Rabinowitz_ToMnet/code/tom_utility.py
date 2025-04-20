import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ----------- Gridworld 環境 -----------
class Gridworld:
    def __init__(self, size=11, object_count=4):
        self.size = size
        self.object_count = object_count
        self.grid = np.zeros((size, size))
        self.agent_pos = (0, 0)
        self.objects = {}
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self.agent_pos = (random.randint(0, self.size-1),
                          random.randint(0, self.size-1))
        self.objects = {}
        for i in range(self.object_count):
            while True:
                pos = (random.randint(0, self.size-1),
                       random.randint(0, self.size-1))
                if pos != self.agent_pos and pos not in self.objects:
                    self.objects[pos] = i + 1
                    break
        return self.get_state()

    def get_state(self):
        state = np.zeros((3, self.size, self.size))
        for (x, y), obj_id in self.objects.items():
            state[0, x, y] = obj_id
        state[1, self.agent_pos[0], self.agent_pos[1]] = 1
        return state

    def step(self, action):
        dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0),
                  3: (1, 0), 4: (0, 0)}[action]
        new_x = min(max(self.agent_pos[0] + dx, 0), self.size - 1)
        new_y = min(max(self.agent_pos[1] + dy, 0), self.size - 1)
        self.agent_pos = (new_x, new_y)
        reward = 0
        done = False
        if self.agent_pos in self.objects:
            reward = 1
            done = True
        return self.get_state(), reward, done

# ----------- 隨機 Agent -----------
class RandomAgent:
    def __init__(self, alpha=0.1):
        self.pi = np.random.dirichlet([alpha] * 5)

    def reset(self):
        pass

    def observe(self, state):
        return state

    def act(self, obs):
        return np.random.choice(5, p=self.pi)

# ----------- Dataset 結構 -----------
class ToMDataset(Dataset):
    def __init__(self, num_episodes=1000, alpha=0.01):
        self.data = []
        env = Gridworld()
        for _ in range(num_episodes):
            agent = RandomAgent(alpha)
            state = env.reset()
            agent.reset()
            obs = agent.observe(state)
            action = agent.act(obs)
            query_state = env.get_state()
            self.data.append((state, action, query_state))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action, query = self.data[idx]
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = F.one_hot(torch.tensor(action), 5).float()
        query_tensor = torch.tensor(query, dtype=torch.float32)
        label = torch.tensor(action, dtype=torch.long)
        return state_tensor, action_tensor, query_tensor, label
    
class ToMDatasetFlexible(Dataset):
    def __init__(self, num_episodes=1000, alpha=0.01, npast=1):
        self.data = []
        self.npast = npast
        env = Gridworld()

        for _ in range(num_episodes):
            agent = RandomAgent(alpha)
            agent.reset()

            past_states, past_actions = [], []
            for _ in range(npast):
                state = env.reset()
                obs = agent.observe(state)
                action = agent.act(obs)
                past_states.append(state)
                past_actions.append(F.one_hot(torch.tensor(action), 5).numpy())

            query_state = env.reset()
            query_obs = agent.observe(query_state)
            query_action = agent.act(query_obs)

            self.data.append((np.stack(past_states), np.stack(past_actions), query_state, query_action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        past_states, past_actions, query_state, action = self.data[idx]
        past_states_tensor = torch.tensor(past_states, dtype=torch.float32)      # (T, C, H, W)
        past_actions_tensor = torch.tensor(past_actions, dtype=torch.float32)    # (T, 5)
        query_tensor = torch.tensor(query_state, dtype=torch.float32)
        label = torch.tensor(action, dtype=torch.long)

        # # 若為單步輸入，則補 T 維度為 1，確保維度一致性
        # if past_states_tensor.dim() == 4:
        #     past_states_tensor = past_states_tensor.unsqueeze(0)
        #     past_actions_tensor = past_actions_tensor.unsqueeze(0)

        return past_states_tensor, past_actions_tensor, query_tensor, label

class ToMMixtureDataset(Dataset):
    def __init__(self, num_episodes=1000, alpha_low=0.01, alpha_high=3.0, ratio=0.5, npast=5):
        self.data = []
        env = Gridworld()

        for _ in range(num_episodes):
            # 根據 ratio 選擇 agent 族群
            alpha = alpha_low if np.random.rand() < ratio else alpha_high
            agent = RandomAgent(alpha)
            agent.reset()

            past_states = []
            past_actions = []

            for _ in range(npast):
                state = env.reset()
                obs = agent.observe(state)
                action = agent.act(obs)
                past_states.append(state)
                past_actions.append(F.one_hot(torch.tensor(action), 5).numpy())

            query_state = env.reset()
            query_obs = agent.observe(query_state)
            query_action = agent.act(query_obs)

            self.data.append((np.stack(past_states), np.stack(past_actions), query_state, query_action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        past_states, past_actions, query_state, action = self.data[idx]
        past_states_tensor = torch.tensor(past_states, dtype=torch.float32)      # (T, C, H, W)
        past_actions_tensor = torch.tensor(past_actions, dtype=torch.float32)    # (T, 5)
        query_tensor = torch.tensor(query_state, dtype=torch.float32)
        label = torch.tensor(action, dtype=torch.long)
        return past_states_tensor, past_actions_tensor, query_tensor, label


# ----------- ToMnet 結構 -----------
class CharacterNet(nn.Module):
    def __init__(self, state_channels=3, action_dim=5, embedding_dim=2):
        super().__init__()
        self.conv = nn.Conv2d(state_channels + action_dim, 8, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, embedding_dim)

    # def forward(self, state, action):
    #     B = state.size(0)
    #     action_spatial = action.view(B, 5, 1, 1).expand(-1, -1, 11, 11)
    #     x = torch.cat([state, action_spatial], dim=1)
    #     x = self.relu(self.conv(x))
    #     x = self.pool(x).view(B, -1)
    #     return self.fc(x)
    
    def forward(self, state, action):
        """
        state:  (B, T, C, H, W)
        action: (B, T, 5)
        """
        try:
            B, T, C, H, W = state.shape
            state = state.view(B * T, C, H, W)
            action = action.view(B * T, 5, 1, 1).expand(-1, -1, H, W)

            x = torch.cat([state, action], dim=1)  # (B*T, C+5, H, W)
            x = self.relu(self.conv(x))            # (B*T, 8, H, W)
            x = self.pool(x).view(B, T, -1)        # (B, T, 8)
            x = x.mean(dim=1)
                                # (B, 8) → 平均所有 Npast episodes
            e_char = self.fc(x)
            return e_char                      # (B, embedding_dim)
        except:
            B = state.size(0)
            action_spatial = action.view(B, 5, 1, 1).expand(-1, -1, 11, 11)
            x = torch.cat([state, action_spatial], dim=1)
            x = self.relu(self.conv(x))
            x = self.pool(x).view(B, -1)
            return self.fc(x)

class PredictionNet(nn.Module):
    def __init__(self, state_channels=3, embedding_dim=2, action_dim=5):
        super().__init__()
        self.conv1 = nn.Conv2d(
            state_channels + embedding_dim, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.fc = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()

    def forward(self, query_state, e_char):
        B = query_state.size(0)
        e_char_spatial = e_char.view(B, -1, 1, 1).expand(-1, -1, 11, 11)
        x = torch.cat([query_state, e_char_spatial], dim=1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(B, -1)
        return self.fc(x)


class ToMnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.char_net = CharacterNet()
        self.pred_net = PredictionNet()

    def forward(self, state, action, query):
        e_char = self.char_net(state, action)
        logits = self.pred_net(query, e_char)
        return logits

# ----------- 訓練程序 -----------
def train_tomnet(dataset, model_name="tomnet"):
    # dataset = ToMDataset(num_episodes=1000, alpha=alpha)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ToMnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(100):
        total_loss = 0
        for state, action, query, label in dataloader:
            logits = model(state, action, query)
            loss = loss_fn(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # 儲存 model 的「參數」
    torch.save(model.state_dict(), f"trained_model\{model_name}.pth")


# ----------- Visualization -----------
def visualize_gridworld_with_history(state_tensor, predicted_action, past_states, past_actions, title="Gridworld with Predicted Action and History"):
    """
    顯示 11x11 Gridworld 並畫出：
    - 🔴 agent 當前位置
    - 🔷 object
    - 🏹 預測動作
    - 🔁 Npast 步歷史行為軌跡（以透明箭頭畫出）
    """
    grid_size = 11
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)

    object_layer = state_tensor[0]
    agent_layer = state_tensor[1]
    agent_pos = None

    # 畫出 objects 和 agent（目前狀態）
    for x in range(grid_size):
        for y in range(grid_size):
            obj_id = object_layer[x, y]
            if obj_id > 0:
                rect = patches.Rectangle((y, grid_size - 1 - x), 1, 1, linewidth=1,
                                         edgecolor='black', facecolor='skyblue')
                ax.add_patch(rect)
                ax.text(y + 0.5, grid_size - 1 - x + 0.5, f'{int(obj_id)}',
                        ha='center', va='center', fontsize=12, color='black')

            if agent_layer[x, y] == 1:
                circ = patches.Circle(
                    (y + 0.5, grid_size - 1 - x + 0.5), 0.3, color='red')
                ax.add_patch(circ)
                agent_pos = (x, y)

    # 畫出預測動作（ToMnet 推論結果）
    if agent_pos is not None:
        x, y = agent_pos
        cx, cy = y + 0.5, grid_size - 1 - x + 0.5

        directions = {
            0: (-0.8, 0),  # left
            1: (0.8, 0),   # right
            2: (0, 0.8),   # up
            3: (0, -0.8),  # down
            4: (0, 0),     # stay
        }

        dx, dy = directions.get(predicted_action, (0, 0))

        if predicted_action == 4:
            ax.text(cx, cy, "✳", fontsize=16, ha='center',
                    va='center', color='orange')
        else:
            ax.arrow(cx, cy, dx, dy, head_width=0.3,
                     head_length=0.3, fc='orange', ec='orange')

    # 畫出歷史動作軌跡（透明箭頭）
    directions_delta = {
        0: (0, -1),  # left
        1: (0, 1),   # right
        2: (-1, 0),  # up
        3: (1, 0),   # down
        4: (0, 0),   # stay
    }

    for state, action in zip(past_states, past_actions):
        agent_layer = state[1]
        agent_pos = np.argwhere(agent_layer == 1)
        if len(agent_pos) == 0:
            continue
        x, y = agent_pos[0]
        cx, cy = y + 0.5, grid_size - 1 - x + 0.5
        dx, dy = directions_delta.get(action, (0, 0))

        if action == 4:
            ax.text(cx, cy, "✳", fontsize=16, ha='center',
                    va='center', color='gray', alpha=0.5)
        else:
            ax.arrow(cx, cy, dy * 0.6, -dx * 0.6, head_width=0.15, head_length=0.15,
                     fc='gray', ec='gray', alpha=0.5, linestyle='--')

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def visualize_gridworld_with_action(state_tensor, predicted_action, title="Gridworld with Predicted Action"):
    """
    顯示 11x11 Gridworld 並畫出 agent 與 object 位置，以及 ToMnet 預測的動作方向
    """
    grid_size = 11
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(range(grid_size + 1))
    ax.set_yticks(range(grid_size + 1))
    ax.grid(True)

    object_layer = state_tensor[0]
    agent_layer = state_tensor[1]

    agent_pos = None

    # 畫出物件與 agent
    for x in range(grid_size):
        for y in range(grid_size):
            obj_id = object_layer[x, y]
            if obj_id > 0:
                rect = patches.Rectangle((y, grid_size - 1 - x), 1, 1, linewidth=1,
                                         edgecolor='black', facecolor='skyblue')
                ax.add_patch(rect)
                ax.text(y + 0.5, grid_size - 1 - x + 0.5, f'{int(obj_id)}',
                        ha='center', va='center', fontsize=12, color='black')

            if agent_layer[x, y] == 1:
                circ = patches.Circle(
                    (y + 0.5, grid_size - 1 - x + 0.5), 0.3, color='red')
                ax.add_patch(circ)
                agent_pos = (x, y)

    # 加入預測動作箭頭（從 agent_pos 開始）
    if agent_pos is not None:
        x, y = agent_pos
        cx, cy = y + 0.5, grid_size - 1 - x + 0.5

        # 預測動作方向對應：←=0, →=1, ↑=2, ↓=3, ·=4
        directions = {
            0: (-0.8, 0),  # left
            1: (0.8, 0),   # right
            2: (0, 0.8),   # up
            3: (0, -0.8),  # down
            4: (0, 0),     # stay
        }

        dx, dy = directions.get(predicted_action, (0, 0))

        if predicted_action == 4:
            # 若為 stay，畫一個 ✳ 圖示在 agent 上
            ax.text(cx, cy, "✳", fontsize=16, ha='center',
                    va='center', color='orange')
        else:
            ax.arrow(cx, cy, dx, dy, head_width=0.3,
                     head_length=0.3, fc='orange', ec='orange')

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()