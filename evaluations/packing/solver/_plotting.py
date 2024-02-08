# plot the solution
import matplotlib.patches as patches

from ._instance import Instance, Solution


def plot_solution(ax, instance: Instance, solution: Solution):
    ax.set_xlim(0, instance.container.width)
    ax.set_ylim(0, instance.container.height)
    for i, box in enumerate(instance.rectangles):
        placement = solution.placements[i]
        if placement:
            ax.add_patch(
                patches.Rectangle(
                    (placement.x, placement.y),
                    box.width if not placement.rotated else box.height,
                    box.height if not placement.rotated else box.width,
                    facecolor="blue" if placement.rotated else "orange",
                    alpha=0.2,
                    edgecolor="b" if placement.rotated else "orange",
                )
            )
    # uniform axis
    ax.set_aspect("equal", adjustable="box")
