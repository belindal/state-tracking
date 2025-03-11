import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import List, Dict, Any


class VisualizationManager:
    """Class for visualization of model analysis results."""
    
    def __init__(self):
        """Initialize visualization settings."""
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
    
    @staticmethod
    def format_subplot(ax, grid_x=True):
        """Format subplot for consistent styling."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if grid_x:
            ax.grid(linestyle='--', alpha=0.4)
        else:
            ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    def plot_probes(self, layerwise_type_scores, plot_name=None):
        """Plot probe accuracy across layers."""
        if plot_name is None:
            return
        
        plt.figure(figsize=(4.5, 5))
        TEXT_FONTSIZE = 16
        
        # Plot with seaborn color palette
        palette = sns.color_palette()
        
        for i, score_type in enumerate(layerwise_type_scores):
            plt.plot(layerwise_type_scores[score_type], label=score_type, 
                     color=palette[i], linewidth=2)
        
        # Set axis labels and title
        plt.xlabel("Layer", fontsize=TEXT_FONTSIZE)
        plt.ylabel("Accuracy", fontsize=TEXT_FONTSIZE)
        plt.title("Probe Accuracy Across Layers", fontsize=TEXT_FONTSIZE, pad=10)
        
        # Set legend
        plt.legend(frameon=True, fancybox=True, framealpha=0.8, fontsize=TEXT_FONTSIZE)
        
        # Set grid and ticks
        plt.grid(True, alpha=0.3)
        plt.xticks(range(len(layerwise_type_scores[list(layerwise_type_scores.keys())[0]])), 
                   range(len(layerwise_type_scores[list(layerwise_type_scores.keys())[0]])), 
                   rotation=90, fontsize=TEXT_FONTSIZE)
        plt.yticks(fontsize=TEXT_FONTSIZE)
        
        # Set axis limits for consistent spacing
        plt.ylim(0, 1.05)
        
        # Create directories and save figure
        os.makedirs("figures/probes", exist_ok=True)
        os.makedirs(os.path.join("figures/probes", os.path.split(plot_name)[0]), exist_ok=True)
        plt.tight_layout()
        plt.savefig(f"figures/probes/{plot_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to figures/probes/{plot_name}.png")
    
    def plot_length_probe_heatmap(self, probe_results, layer_names, plot_name):
        """Plot heatmap of probe accuracies across lengths and layers."""        
        # Convert results to matrix form
        lengths = sorted(probe_results.keys())
        assert len(probe_results[lengths[0]]) == len(layer_names)
        cmap = sns.color_palette("viridis", as_cmap=True)

        accuracy_matrix = np.zeros((len(layer_names), len(lengths)))
        variance_matrix = np.zeros((len(layer_names), len(lengths)))
        for i, layer in enumerate(layer_names):
            for j, length in enumerate(lengths):
                accuracy_matrix[i, j] = probe_results[length][i].mean()
                variance_matrix[i, j] = probe_results[length][i].var()
        
        # Create heatmap with consistent styling
        plt.figure(figsize=(0.1 * len(lengths) + 2, 0.15 * len(layer_names) + 2))
        TEXT_FONTSIZE = 16
        
        sns.heatmap(
            accuracy_matrix,
            xticklabels=[length for length in lengths if length % 5 == 0],
            yticklabels=layer_names,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Accuracy'}
        )
        
        # Set ticks
        plt.xticks(ticks=[i + 0.5 for i in range(len(lengths)) if i % 5 == 0], 
                   labels=[length for length in lengths if length % 5 == 0],
                   fontsize=TEXT_FONTSIZE)
        plt.yticks(fontsize=TEXT_FONTSIZE)
        
        # Set axis labels and title
        plt.xlabel('Sequence Length', fontsize=TEXT_FONTSIZE)
        plt.ylabel('Layer', fontsize=TEXT_FONTSIZE)
        plt.title('Probe Accuracy by Layer and Sequence Length', fontsize=TEXT_FONTSIZE, pad=10)
        
        # Create directories and save figure
        os.makedirs('figures/lengthwise_probe', exist_ok=True)
        os.makedirs(os.path.join("figures/lengthwise_probe", os.path.split(plot_name)[0]), exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'figures/lengthwise_probe/{plot_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved lengthwise probe heatmap to figures/lengthwise_probe/{plot_name}.png")
    
    def plot_logits(self, logit_diff, token_positions, layer_names, plot_name=None):
        """Plot logit differences from activation patching."""

        cmap = sns.blend_palette([
            "white", "#7EB4F8", "#70AAF2", "#64A0EB", "#2965A5"
        ], n_colors=10)
        
        plt.figure(figsize=(0.1 * len(token_positions) + 2, 0.15 * len(layer_names) + 2))
        ax = sns.heatmap(
            np.array(logit_diff, dtype=float), annot=None,
            fmt='', cmap=cmap, linewidths=.5,
            linecolor='black',
        )
        cbar = ax.collections[0].colorbar
        cbar.set_label('Logit Difference', fontsize=20)
        # set ticks range to [0,1]
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        cbar.ax.tick_params(labelsize=20)
        
        plt.xlabel('Input Tokens', fontsize=20)  # Increased font size
        plt.ylabel('Layers', fontsize=20)  # Increased font size
        
        plt.yticks(np.arange(len(layer_names)) + 0.5, layer_names, rotation=0, fontsize=20)  # Increased font size
        plt.gca().xaxis.tick_top()
        plt.gca().xaxis.set_label_position("top")
        plt.xticks(np.arange(0, len(token_positions), 5) + 0.5, token_positions[::5], rotation=90, fontsize=20)  # Show intervals
        plt.tight_layout()
        os.makedirs(os.path.join("figures/intervene", os.path.split(plot_name)[0]), exist_ok=True)
        self.format_subplot(plt.gca())
        plt.savefig(f"figures/intervene/{plot_name}.png")
        plt.close()
    

    def format_subplot(self, ax, grid_x=True):
        """Format subplot for consistent styling."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if grid_x:
            ax.grid(linestyle='--', alpha=0.4)
        else:
            ax.grid(axis='y', linestyle='--', alpha=0.4)

