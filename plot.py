import matplotlib.pyplot as plt
import numpy as np

line_thickness = 1


# --------- Plot Accuracy ---------
plt.figure(figsize=(6, 4))
colors = ["#09ef2f", "#3709ef", 'black', "#ff0000"]
legend_names = ['DCT', 'PCA', 'Raw', 'Drift']
for idx, item in enumerate(histories):
    history = item['history']
    label_base = legend_names[idx]
    color = colors[idx % len(colors)]
    
    # Multiply by 100 to convert to percentage
    plt.plot(np.array(history['accuracy']) * 100, label=f'{label_base} Train', color=color, linestyle='-', linewidth=line_thickness)
    plt.plot(np.array(history['val_accuracy']) * 100, label=f'{label_base} Val', color=color, linestyle='--', linewidth=line_thickness)

plt.title('Training and Validation Accuracy', fontsize=12, weight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Accuracy (%)', fontsize=10)  # <-- Update label
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('accuracy_curves.png', dpi=1000)
plt.show()


# --------- Plot Loss ---------
plt.figure(figsize=(6, 4))
for idx, item in enumerate(histories):
    history = item['history']
    label_base = legend_names[idx]
    color = colors[idx % len(colors)]
    plt.plot(history['loss'], label=f'{label_base} Train', color=color, linestyle='-', linewidth=line_thickness)
    plt.plot(history['val_loss'], label=f'{label_base} Val', color=color, linestyle='--', linewidth=line_thickness)
plt.title('Training and Validation Loss', fontsize=12, weight='bold')
plt.xlabel('Epoch', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=1000)
plt.show()

# --------- Plot Actual Preparation and Training Times ---------
# Assuming `prep_times` and `train_times` are dictionaries with keys matching legend_names

plt.figure(figsize=(12, 7))
bar_width = 0.4
index = np.arange(len(legend_names))
prep_vals = [prep_times[name] for name in legend_names]
train_vals = [train_times[name] for name in legend_names]
prep_max = max(prep_vals)
train_max = max(train_vals)
# Plot actual times
bars_pre = plt.bar(index, prep_vals, bar_width, label='Preparation Time', color="#03f614", alpha=0.8, linewidth=line_thickness)
bars_tr = plt.bar(index + bar_width, train_vals, bar_width, label='Training Time', color="#1109ef", alpha=0.8, linewidth=line_thickness)

for bar in bars_pre:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars_tr:
    height = bar.get_height()
    plt.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 4),
                 textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Preparation and Training Times (Actual)', fontsize=16, fontweight='bold')
plt.xlabel('Feature Set', fontsize=14, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=14, fontweight='bold')
plt.xticks(index + bar_width/2, legend_names)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
plt.savefig('actual_times.png', dpi=1000)
plt.show()

# --------- Plot Normalized Times ---------
# Normalize to max for train and prep times separately
train_vals = np.array([train_times[name] for name in legend_names])
prep_vals = np.array([prep_times[name] for name in legend_names])

train_baseline_idx = np.argmax(train_vals)
prep_baseline_idx = np.argmax(prep_vals)

train_baseline = train_vals[train_baseline_idx]
prep_baseline = prep_vals[prep_baseline_idx]

train_norm = train_vals / train_baseline
prep_norm = prep_vals / prep_baseline

plt.figure(figsize=(12, 7))
bars_pre = plt.bar(index, prep_norm, bar_width, label='Prep (Normalized)', color="#03f614", alpha=0.8, linewidth=line_thickness)
bars_tr = plt.bar(index + bar_width, train_norm, bar_width, label='Train (Normalized)', color="#1109ef", alpha=0.8, linewidth=line_thickness)

for bar in bars_pre:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0,4), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars_tr:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0,4), textcoords='offset points', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Normalized Preparation and Training Times', fontsize=16, fontweight='bold')
plt.xlabel('Feature Set', fontsize=14, fontweight='bold')
plt.ylabel('Normalized Time', fontsize=14, fontweight='bold')
plt.xticks(index + bar_width/2, legend_names)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5, axis='y')
plt.tight_layout()
plt.show()
