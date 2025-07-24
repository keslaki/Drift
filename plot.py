# this is the same plot code for both MNIST and CIFAR100. Change the saved name at the end

import pickle

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

# ======= Parameters =======
inset = 'on'  # Set to 'on' to show insets, 'off' to hide insets

LINE_THICKNESS = 1  # Line width
INSET_EPOCH_START = 50
INSET_EPOCH_END = 200
INSET_POSITION = [0.25, 0.4, 0.4, 0.4]  # [left, bottom, width, height]

# Inset font size controls
INSET_TICK_LABELSIZE = 8
INSET_TITLE_FONTSIZE = 10
INSET_LABEL_FONTSIZE = 9

# ======= Load Data =======
with open('training_histories.pkl', 'rb') as f:
    data = pickle.load(f)
    all_methods_epoch_histories = data['histories']
    CONFIG = data['CONFIG']

epoch_range = range(1, CONFIG['epochs'] + 1)

# ======= Plot Settings =======
line_styles_train = ['-', '-', '-', '-']
line_styles_val = ['-.', '-.', '-.', '-.']
colors = [
    "#1aff01", "#002afa", "#000000", "#ff0000", '#9467bd', '#8c564b',
    '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Expand color palette if needed
if len(all_methods_epoch_histories) > len(colors):
    import matplotlib.cm as cm
    colors += [cm.get_cmap('tab20')(i) for i in range(len(all_methods_epoch_histories) - len(colors))]

# ======= Plot Function =======
def plot_metric_with_inset(metric_name, ylabel, filename, legend_loc='upper right'):
    fig, ax = plt.subplots(figsize=(6, 4))
    #ax.set_title(f'{metric_name} vs. Epochs', fontsize=16)
    #ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)

    for i, (method_name, histories_list) in enumerate(all_methods_epoch_histories.items()):
        history = histories_list[0]
        color = colors[i % len(colors)]

        ax.plot(epoch_range, history[metric_name],
                label=f'{method_name} (Train)',
                color=color,
                linestyle=line_styles_train[i % len(line_styles_train)],
                linewidth=LINE_THICKNESS,
                alpha=0.8)

        ax.plot(epoch_range, history[f'val_{metric_name}'],
                label=f'{method_name} (Validation)',
                color=color,
                linestyle=line_styles_val[i % len(line_styles_val)],
                linewidth=LINE_THICKNESS,
                alpha=0.8)

    ax.legend(loc=legend_loc, fontsize=8)
    ax.autoscale()
    plt.tight_layout()

    # Add inset if enabled
    if inset == 'on':
        ax_inset = fig.add_axes(INSET_POSITION)
        ax_inset.set_title("", fontsize=INSET_TITLE_FONTSIZE)
        ax_inset.grid(True, linestyle=':', alpha=0.5)

        inset_range = list(range(INSET_EPOCH_START, min(INSET_EPOCH_END + 1, CONFIG['epochs'] + 1)))

        for i, (method_name, histories_list) in enumerate(all_methods_epoch_histories.items()):
            history = histories_list[0]
            color = colors[i % len(colors)]

            ax_inset.plot(inset_range,
                          history[metric_name][INSET_EPOCH_START - 1:INSET_EPOCH_END],
                          color=color,
                          linestyle=line_styles_train[i % len(line_styles_train)],
                          linewidth=LINE_THICKNESS * 1,
                          alpha=0.8)

            ax_inset.plot(inset_range,
                          history[f'val_{metric_name}'][INSET_EPOCH_START - 1:INSET_EPOCH_END],
                          color=color,
                          linestyle=line_styles_val[i % len(line_styles_val)],
                          linewidth=LINE_THICKNESS * 1,
                          alpha=0.8)

        ax_inset.tick_params(axis='both', labelsize=INSET_TICK_LABELSIZE)
        ax_inset.set_xlabel('', fontsize=INSET_LABEL_FONTSIZE)
        ax_inset.set_ylabel(ylabel, fontsize=INSET_LABEL_FONTSIZE)

    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    plt.show()

# ======= Call Plot Functions =======
plot_metric_with_inset('accuracy', '', 'accuracy_with_inset.png', legend_loc='lower right')
plot_metric_with_inset('loss', '', 'loss_with_inset.png', legend_loc='upper right')
