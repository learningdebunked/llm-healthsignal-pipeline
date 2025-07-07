import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

# Simulated predictions and ground truths for 4 ablation configurations
# In practice, replace these with actual model outputs
ground_truth = ['N', 'V', 'N', 'AF', 'N', 'AF', 'N', 'V', 'N', 'N']

# A: LSTM only
preds_A = ['N', 'V', 'N', 'N', 'N', 'N', 'N', 'V', 'N', 'N']

# B: GAN + LSTM
preds_B = ['N', 'V', 'N', 'AF', 'N', 'AF', 'N', 'V', 'N', 'N']

# C: GAN + LSTM + GPT-2
preds_C = ['N', 'V', 'N', 'AF', 'N', 'AF', 'N', 'V', 'N', 'N']

# D: LSTM + GPT-2 (no GAN)
preds_D = ['N', 'V', 'N', 'N', 'N', 'AF', 'N', 'V', 'N', 'N']

# Optional LLM survey scores (out of 5)
llm_scores = {
    'A': None,
    'B': None,
    'C': 4.6,
    'D': 4.5
}

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    return acc, f1

# Evaluate all configs
configs = ['A (LSTM)', 'B (GAN + LSTM)', 'C (Full GenAI)', 'D (LSTM + GPT-2)']
results = []

for config, preds in zip(configs, [preds_A, preds_B, preds_C, preds_D]):
    acc, f1 = compute_metrics(ground_truth, preds)
    score = llm_scores[config[0]]
    results.append({'Config': config, 'Accuracy': acc, 'F1 Score': f1, 'LLM Helpfulness': score})

# Save results
df = pd.DataFrame(results)
df.to_csv("/mnt/data/ablation_results.csv", index=False)
print(df)

# Plot
plt.figure(figsize=(10, 5))
bar_width = 0.35
x = np.arange(len(df))

plt.bar(x - bar_width/2, df['Accuracy'], width=bar_width, label='Accuracy')
plt.bar(x + bar_width/2, df['F1 Score'], width=bar_width, label='F1 Score')

plt.xticks(x, df['Config'], rotation=15)
plt.ylabel("Score")
plt.title("Ablation Study - Accuracy and F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/data/ablation_performance.png")
plt.show()