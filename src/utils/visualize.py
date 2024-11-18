import matplotlib.pyplot as plt
import os
import numpy as np

def plot_model_loss(final_epoch_losses, output_file="model-loss.png"):
    # Create the plot
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(final_epoch_losses) + 1))

    # Plot the training loss
    plt.plot(epochs, final_epoch_losses, marker='o', label='Training Loss', linewidth=2)

    # Highlight the minimum loss
    min_loss = min(final_epoch_losses)
    min_epoch = final_epoch_losses.index(min_loss) + 1
    plt.scatter(min_epoch, min_loss, color='red', s=100, label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
    plt.text(min_epoch, min_loss, f'{min_loss:.4f}', color='red', fontsize=10, ha='center', va='bottom')

    # Customize the plot
    plt.title("Model Loss Per Epoch", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.xticks(epochs, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12)

    # Save the plot
    plt.tight_layout()

    save_path = "plots/" + output_file
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close()

def plot_test_mean_auc(final_results_df):
    tasks = final_results_df["Task"]
    auc_scores = final_results_df["AUC"]

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(tasks))  # Position for each task on the y-axis

    # Create bars
    bars = ax.barh(y_pos, auc_scores, color='skyblue', edgecolor='black')

    # Add text labels to the bars
    for bar, auc in zip(bars, auc_scores):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f"{auc:.3f}", va='center', fontsize=10)

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tasks, fontsize=12)
    ax.set_xlabel("AUC Score", fontsize=14)
    ax.set_ylabel("Task", fontsize=14)
    ax.set_title("Test AUC for Each Task (Horizontal)", fontsize=16)
    ax.set_xlim(0, 1)  # Set x-axis range to [0, 1]

    # Style adjustments
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.show()
