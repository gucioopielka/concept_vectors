import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_RDM_concept(rdm, rel_indices=None, rel_ticks=True, title='RDM'):

    # Plot the RDM
    plt.imshow(1-rdm, cmap='coolwarm')
    plt.title(title, fontsize=16)

    if rel_indices:
        for analogy, (start, end) in rel_indices.items():
            width = height = end - start

            # Draw a rectangle
            rect = patches.Rectangle((start, start),
                                    width,
                                    height,
                                    linewidth=0.8,
                                    edgecolor='purple',
                                    facecolor='none')
            plt.gca().add_patch(rect)

        midpoints = [(start + end) / 2 for start, end in rel_indices.values()]

        if rel_ticks:
            plt.xticks(midpoints)
            plt.yticks(midpoints)
            plt.gca().set_xticklabels(rel_indices.keys(), rotation=90, fontsize=10)
            plt.gca().set_yticklabels(rel_indices.keys(), fontsize=10)

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Similarity', fontsize=12)
    cbar.ax.tick_params(labelsize=10)


    plt.tight_layout()
    plt.show()
