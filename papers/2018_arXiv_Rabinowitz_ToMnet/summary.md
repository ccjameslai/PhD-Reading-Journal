# 📘 Machine Theory of Mind (ToMnet)

- **Authors**: Rabinowitz et al.
- **Affiliation**: DeepMind
- **Year**: 2018
- **Venue**: arXiv preprint
- **Link**: [arXiv:1802.07740](https://arxiv.org/abs/1802.07740)
- **Subject**: 從觀察行為推論心智狀態的神經網路模型
- **Keywords**: Theory of Mind, Meta-Learning, Agent Modeling, Belief Inference

------

## 🟨 一、研究動機與貢獻

### ❓為什麼需要 Machine ToM？

- 人類能**快速從少量觀察中推論他人目標與信念**
- 現有 AI 難以處理含**錯誤信念**或**行為不一致**的社會互動場景

### ✅ 本文貢獻

- 提出 **ToMnet 架構**：透過 **meta-learning** 學會建模他人行為
- 不依賴明確 reward、belief label，**純觀察就能推理他人目標與錯誤信念**
- 重現心理學經典實驗（如 Sally-Anne Test）

------

## 🧩 二、ToMnet 架構與訓練流程

![architecture_diagram](/papers/2018_arXiv_Rabinowitz_ToMnet/architecture_diagram.png)

| 模組               | 功能                                          |
| ------------------ | --------------------------------------------- |
| **CharacterNet**   | 根據過去行為 episode 萃取 agent 嵌入 `e_char` |
| **MentalStateNet** | 根據目前行為片段推測當下心理狀態 `e_mental`   |
| **PredictionNet**  | 預測下一步行為、物件消耗、SR、belief 狀態     |

- 損失函數：行為 NLL + belief CE + SR + DVIB
- 訓練方式：給定 N_past 筆軌跡，預測新環境中的行為與 belief

------

## 🔬 三、五大實驗與對應能力

| 實驗段落          | 展示能力                                   | 對應心理概念                         |
| ----------------- | ------------------------------------------ | ------------------------------------ |
| 3.1 隨機 agent    | Bayes-optimal 行為推理                     | 頻率統計與先驗建構                   |
| 3.2 有偏好 agent  | 推論 agent reward 偏好與目標               | 目標導向行為辨識 (Woodward, 1998)    |
| 3.3 deep RL agent | 分辨 agent species 並預測行為              | 社會推理與泛化                       |
| 3.4 False Belief  | 即使看不到世界變化，仍據錯誤信念行動       | Sally-Anne Test (Baron-Cohen, 1985)  |
| 3.5 顯式信念預測  | 能預測 agent 的 belief map（物件位置認知） | 可視能力與信念形成（meta-cognition） |

------

## 🧬 四、理論意涵與心理學對應

- ToMnet = 一個學會「**我知道他不知道**」的推理機器
- 不需要 symbolic 表徵就能完成 belief 推理
- 換句話說：**透過觀察別人「錯誤的行為」，ToMnet 學會「他有錯誤的信念」**

------

## 🧠 五、My Thoughts

### ✅ Why I Chose This Paper

This is one of the foundational works that bridges cognitive science concepts (Theory of Mind) with meta-learning and agent modeling. It aligns strongly with my PhD direction: **robotic inference of human intent in uncertain environments**. Moreover, I strongly desire to know how to implement ToM with algorithms.

### ✅ Key Inspiration Points

- Neural network as meta-theory of other agents
- Compact agent embedding (`e_char`) → possible to condition other models on it
- Predicting false belief is a fascinating cognitive alignment

### ✅ Possible Pitfalls

- Learned beliefs are **implicit**: lacks interpretability
- CharacterNet struggles if input sequence is too short or too noisy

### ✅ 若聚焦在人機協作：

- 在低資料、人類行為重複性高的場景中，**ToMnet 可作為策略預測器**
- 可結合 ToMnet 與模仿學習（IL）以強化 intent inference 能力
- 設計 agent→user 語言回饋：**ToMnet + belief verbalization** = 可解釋 AI

### ✅ 若聚焦於 ToM 理論擴展：

- 加入 recursive ToM：我知道他正在推測我（second-order belief）
- 建立 belief embedding 空間，可查詢他人對任務相關變數的信念
- 建立少樣本、對話式 belief 推理架構，並將語言與行為整合

### ✅ Future Ideas

1. **ToMnet 是否適合用於我的人機協作場景？**（動作變異性低但資料少）
2. 如果我要讓 AI 解釋自己的 belief 推論，需要怎樣的表徵方式？
3. 有無可能將 ToMnet 與 LLM 結合，做信念推理 + 策略生成？
4. 如何針對錯誤信念進行對話式修正？這是否可應用於教學機器人？
5. 若做 cross-agent generalization，如何處理 belief update 的轉移問題？
6. Extend ToMnet to **multi-modal observation**: vision + trajectory
7. Use e_char as a conditioning factor in **collaborative policy learning**
8. Embed ToMnet-style belief inference into **inverse RL** pipeline

