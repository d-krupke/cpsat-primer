from typing import Hashable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.patches import Patch
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def plot_cvrp_solution(
    G: nx.Graph,
    depot: Hashable,
    tours: list[list[Hashable]],
    vehicle_capacity: int | None = None,
    figsize: tuple[int, int] = (14, 10),
    title: str | None = None,
    show_demands: bool = True,
    show_loads: bool = True,
    show_objective: bool = True,
    node_size: int = 100,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Enhanced visualization of CVRP solution with route-specific coloring,
    demand information shown by node size and color intensity, and route statistics.

    Parameters:
        G (nx.Graph): Graph with nodes, edges, positions and demands
        depot (Any): Identifier for the depot node
        tours (list): List of tours, each tour is a list of nodes (excluding depot)
        vehicle_capacity (int, optional): Vehicle capacity constraint
        figsize (tuple): Figure size
        title (str, optional): Title for the plot
        show_demands (bool): Whether to visually encode demand values with node size/color
        show_loads (bool): Whether to display route load information
        show_objective (bool): Whether to calculate and display the objective value
        node_size (int): Base size of node markers
        show_legend (bool): Whether to display the legend

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Get positions and demand attributes
    pos = nx.get_node_attributes(G, "pos")
    demands = nx.get_node_attributes(G, "demand")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Available colors for the routes (colorblind-friendly palette)
    route_colors = list(mcolors.TABLEAU_COLORS.values())

    # Create a color map for demand values
    demand_cmap = cm.get_cmap("YlOrRd")  # Yellow-Orange-Red colormap

    # Calculate min and max demands for scaling
    min_demand = min(demands.values())
    max_demand = max(demands.values())

    # Draw the depot
    depot_x, depot_y = pos[depot]
    ax.scatter(
        depot_x,
        depot_y,
        s=node_size * 2.5,
        c="black",
        marker="*",
        edgecolors="black",
        linewidth=1.5,
        zorder=5,
    )

    # Add depot label
    ax.text(
        depot_x,
        depot_y - 0.03,
        "Depot",
        ha="center",
        va="top",
        fontweight="bold",
        zorder=6,
        color="black",
    )

    # Prepare legend for routes
    route_legend_elements: list[Patch] = []

    # Calculate total objective value if requested
    total_distance: float = 0

    # Draw each tour with a different color
    for i, tour in enumerate(tours):
        # Select color for this route
        route_color = route_colors[i % len(route_colors)]

        # Calculate the total demand for this tour
        tour_demand = sum(demands[node] for node in tour)

        # Create full tour including depot
        full_tour = list(tour)
        if full_tour[0] != depot:
            full_tour = [depot] + full_tour
        if full_tour[-1] != depot:
            full_tour.append(depot)

        # Extract coordinates for the route
        route_x = [pos[node][0] for node in full_tour]
        route_y = [pos[node][1] for node in full_tour]

        # Draw the route
        ax.plot(route_x, route_y, c=route_color, linewidth=2.5, alpha=0.7, zorder=2)

        # Calculate route distance if needed
        route_distance: float = 0
        if show_objective:
            for j in range(len(full_tour) - 1):
                u, v = full_tour[j], full_tour[j + 1]
                if G.has_edge(u, v) and "weight" in G[u][v]:
                    route_distance += G[u][v]["weight"]
            total_distance += route_distance

        # Add arrows to show direction
        for j in range(len(full_tour) - 1):
            start_x, start_y = pos[full_tour[j]]
            end_x, end_y = pos[full_tour[j + 1]]
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            dx = end_x - start_x
            dy = end_y - start_y
            arrow_len = np.sqrt(dx**2 + dy**2)
            ax.arrow(
                mid_x - dx / arrow_len * 0.02,
                mid_y - dy / arrow_len * 0.02,
                dx / arrow_len * 0.04,
                dy / arrow_len * 0.04,
                head_width=0.015,
                head_length=0.03,
                fc=route_color,
                ec=route_color,
                zorder=3,
            )

        # Add info about this route to legend
        if show_loads:
            route_label = f"Route {i + 1}: Load {tour_demand}"
            if vehicle_capacity:
                utilization = tour_demand / vehicle_capacity * 100
                route_label += f" ({utilization:.1f}%)"

                # Check if capacity constraints are violated
                if tour_demand > vehicle_capacity:
                    route_label += " ⚠️"  # Warning symbol for overloaded routes

            # Add distance information if available
            if show_objective:
                route_label += f", Distance: {route_distance:.2f}"

            route_legend_elements.append(
                Patch(facecolor=route_color, alpha=0.7, label=route_label)
            )

    # Draw all non-depot nodes with size/color based on demand
    if show_demands:
        demand_legend_elements: list = []

        # Draw all non-depot nodes with different sizes based on demand
        for node in G.nodes():
            if node != depot:
                x, y = pos[node]
                # Scale node size based on demand
                size_factor = 0.5 + 1.5 * (demands[node] - min_demand) / max(
                    1, max_demand - min_demand
                )
                # Get color based on demand
                color_intensity = (demands[node] - min_demand) / max(
                    1, max_demand - min_demand
                )
                node_color = demand_cmap(color_intensity)

                # Draw node
                ax.scatter(
                    x,
                    y,
                    s=node_size * size_factor,
                    c=[node_color],
                    edgecolors="black",
                    linewidth=1,
                    zorder=4,
                )

                # No node labels

        # Create demand legend elements
        legend_demands = range(min_demand, max_demand + 1)
        for d in legend_demands:
            size_factor = 0.5 + 1.5 * (d - min_demand) / max(1, max_demand - min_demand)
            color_intensity = (d - min_demand) / max(1, max_demand - min_demand)
            demand_legend_elements.append(
                plt.scatter(
                    [],
                    [],
                    s=node_size * size_factor,
                    c=[demand_cmap(color_intensity)],
                    edgecolors="black",
                    label=f"Demand: {d}",
                )
            )

    # Set plot properties
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Set title with solution statistics
    if title:
        plt.title(title, fontsize=14, fontweight="bold")
    else:
        num_vehicles = len(tours)
        total_demand = sum(demands.values())
        title_text = (
            f"CVRP Solution - {num_vehicles} Vehicles, Total Demand: {total_demand}"
        )
        if show_objective:
            title_text += f", Total Distance: {total_distance:.2f}"
        plt.title(title_text, fontsize=14, fontweight="bold")

    # Add legends - Fixed to ensure they always show up
    if show_legend:
        # Create figure legend items
        legend_items: list = []
        legend_labels: list[str] = []

        # Add route legend items
        for i, item in enumerate(route_legend_elements):
            legend_items.append(Patch(facecolor=item.get_facecolor(), alpha=0.7))
            legend_labels.append(item.get_label())  # type: ignore
        # Add depot legend item
        legend_items.append(
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="black",
                markersize=10,
                markeredgecolor="black",
            )
        )

        legend_labels.append("Depot")

        # Add demand legend if needed
        if show_demands:
            for d in legend_demands:
                size_factor = 0.5 + 1.5 * (d - min_demand) / max(
                    1, max_demand - min_demand
                )
                legend_items.append(
                    Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor=demand_cmap(color_intensity),  # type: ignore
                        markersize=np.sqrt(size_factor) * 5,
                        markeredgecolor="black",
                    )
                )

                legend_labels.append(f"Demand: {d}")

        # Create the legend
        ax.legend(
            legend_items,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0.5,
            title="Route and Demand Information",
        )

    # Summary statistics
    stats_lines: list[str] = []
    stats_lines.append(f"Number of Routes: {len(tours)}")

    if vehicle_capacity:
        route_loads = [sum(demands[node] for node in tour) for tour in tours]
        max_load = max(route_loads) if route_loads else 0
        min_load = min(route_loads) if route_loads else 0
        avg_load = sum(route_loads) / len(route_loads) if route_loads else 0

        stats_lines.append(f"Vehicle Capacity: {vehicle_capacity}")
        stats_lines.append(
            f"Max Route Load: {max_load} ({max_load / vehicle_capacity * 100:.1f}%)"
        )
        stats_lines.append(
            f"Min Route Load: {min_load} ({min_load / vehicle_capacity * 100:.1f}%)"
        )
        stats_lines.append(
            f"Avg Route Load: {avg_load:.1f} ({avg_load / vehicle_capacity * 100:.1f}%)"
        )

    if show_objective:
        stats_lines.append(f"Total Distance: {total_distance:.2f}")
        if len(tours) > 0:
            avg_distance = total_distance / len(tours)
            stats_lines.append(f"Avg Distance per Route: {avg_distance:.2f}")

    stats_text = "Statistics:\n" + "\n".join(stats_lines)

    plt.figtext(
        0.01,
        0.01,
        stats_text,
        fontsize=10,
        bbox=dict(
            facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round,pad=0.5"
        ),
    )

    # Adjust layout to ensure the legend is visible
    plt.tight_layout(rect=(0, 0, 0.85, 1))  # Make room for the legend

    return fig, ax
