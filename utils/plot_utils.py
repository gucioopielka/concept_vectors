import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_RDM(rdm, 
             axis=None, 
             rel_indices=None, 
             rel_ticks=True, 
             title='RDM',
             norm=None,
    ):
    # Create a new figure and axis if none is provided
    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # Plot the RDM on the specified (or new) axes object
    cax = ax.imshow(1-rdm, cmap='coolwarm', norm=norm)
    ax.set_title(title, fontsize=14)

    if rel_indices:
        for analogy, (start, end) in rel_indices.items():
            width = height = end - start

            # Draw a rectangle on the specified axes
            rect = patches.Rectangle((start, start), width, height, linewidth=0.8, edgecolor='purple', facecolor='none')
            ax.add_patch(rect)

        midpoints = [(start + end) / 2 for start, end in rel_indices.values()]

        if rel_ticks:
            ax.set_xticks(midpoints)
            ax.set_yticks(midpoints)
            ax.set_xticklabels(rel_indices.keys(), rotation=90, fontsize=10)
            ax.set_yticklabels(rel_indices.keys(), fontsize=10)

    # Add colorbar for the axes
    if axis is None:
        cbar = plt.colorbar(cax, ax=ax)
        cbar.ax.set_ylabel('Similarity', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    # Adjust layout
    plt.tight_layout()
    if axis is None:
        plt.show()
