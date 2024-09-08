# module for component code
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch, ArrowStyle
from matplotlib.lines import Line2D
from matplotlib.text import Annotation
# class to act as knowledge object for each component in the system
class Component:
    def __init__(self, name, list_position, correlation_dict= None, p_value_dict= None, non_lin_correlation_dict= None, non_lin_p_value_dict = None ):
        self.name = name
        # position in results list
        self.list_position = list_position
        self.top_n_corrs = 0

        # empty dict for results from EDA for components relationship to all other components
        self.correlation_dict = correlation_dict if correlation_dict is not None else {}
        self.p_value_dict = p_value_dict if p_value_dict is not None else {}
        self.non_lin_correlation_dict = non_lin_correlation_dict if non_lin_correlation_dict is not None else {}
        self.non_lin_p_value_dict = non_lin_p_value_dict if p_value_dict is not None else {}

    def get_correlated_components(self, source_data= 'linear', lower_threshold= 0, upper_threshold= 1):
        """
        Returns correlations, takes optional min threshold argument (default is 1 for all values).

        Parameters:
        - threshold: The minimum correlation value.

        Returns:
        - A dictionary containing component names as keys and correlation values as values.
        """
        correlated_components = {}
        if source_data == 'linear':
            for component, correlation in self.correlation_dict.items():
                correlation = float(correlation)
                if component != self.name and  lower_threshold <= abs(correlation) <= upper_threshold:
                    correlated_components[component] = correlation
            return correlated_components

        if source_data == 'non_linear':
            correlated_components = {}
            for component, correlation in self.non_lin_correlation_dict.items():
                correlation = float(correlation)
                if component != self.name and  lower_threshold <= abs(correlation) <= upper_threshold:
                    correlated_components[component] = correlation
            return correlated_components

    # return n strongerst correlations
    def get_strongest_correlated_components(self, source_data= 'linear', top_n_corrs = 0):
        """
        Returns correlations, takes optional min threshold argument (default is 1 for all values).

        Parameters:
        - top_n_corrs: Number of strongest correlations to return.

        Returns:
        - A dictionary containing component names as keys and correlation values as values.
        """

        if source_data == 'linear':
            sorted_dict = dict(sorted(self.correlation_dict.items(), key=lambda item: item[1], reverse=True))

            return dict(list(sorted_dict.items())[:top_n_corrs])

        if source_data == 'non_linear':
            sorted_dict = dict(sorted(self.non_lin_correlation_dict.items(), key=lambda item: item[1], reverse=True))

            return dict(list(sorted_dict.items())[:top_n_corrs])

    def get_correlated_components_p_value(self, source_data= 'linear', lower_threshold= 0, upper_threshold= 0.05):
        """
        Returns correlated components by p-value, takes optional max threshold argument (default is 1 for all values).

        Parameters:
        - threshold: The minimum correlation value.

        Returns:
        - A dictionary containing component names as keys and p values as values.
        """
        correlated_components_p_value = {}

        if source_data == 'linear':
            for component, p_value in self.p_value_dict.items():
                p_value = float(p_value)
                if component != self.name and  lower_threshold <= p_value <= upper_threshold:
                    correlated_components_p_value[component] = p_value
            return correlated_components_p_value

        if source_data == 'non_linear':
            for component, p_value in self.non_lin_p_value_dict.items():
                p_value = float(p_value)
                if component != self.name and  lower_threshold <= p_value <= upper_threshold:
                    correlated_components_p_value[component] = p_value
            return correlated_components_p_value



def plot_correlation_network(component1,lower_threshold= 0.5, upper_threshold= 1, top_n_corrs = 0 ,layout= nx.spring_layout ):
    """
    Plot a network of components linked by correlation strength above a certain threshold,
    showing an edge if either linear or non-linear correlation is above the minimum threshold,
    and differentiating between correlations with curved edges. Labels are offset to reduce overlap.
    """
    G = nx.MultiDiGraph()
    # top_n_corrs = int(top_n_corrs)

    # Ensure top_n_corrs is an integer
    if not isinstance(top_n_corrs, int):
        raise ValueError("top_n_corrs must be an integer.")
    print(type(top_n_corrs))

    # check if top correlations wanted
    if top_n_corrs > 0:
        # Add nodes (components) and edges ( correlation)
        for component, correlation in component1.get_strongest_correlated_components('linear', top_n_corrs).items():
            G.add_node(component, correlation_strength=correlation)
            G.add_edge(component1.name, component, weight= correlation, correlation_type='Linear')

        for component, correlation in component1.get_strongest_correlated_components('non_linear', top_n_corrs).items():
            G.add_node(component, correlation_strength=correlation)
            G.add_edge(component1.name, component, weight= correlation, correlation_type='Non-linear')

    # otherwise return results within thresholds
    else:
        # Add nodes and edges for linear correlation
        for component, correlation in component1.correlation_dict.items():
            if lower_threshold <= abs(correlation) <= upper_threshold or abs(component1.non_lin_correlation_dict.get(component, 0)) > lower_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Linear')

        # Add nodes and edges for non-linear correlation
        for component, correlation in component1.non_lin_correlation_dict.items():
            if lower_threshold <= abs(correlation) <= upper_threshold or abs(component1.correlation_dict.get(component, 0)) > lower_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Non-linear')

    plt.figure(figsize=(10, 10))
    pos = layout(G)

    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue", alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # Function to draw edges with FancyArrowPatch and labels with offset
    def draw_edges_with_arrows_and_labels():
        for (u, v, attribs) in G.edges(data=True):
            edge_type = attribs['correlation_type']
            weight = attribs['weight']
            color = 'red' if edge_type == 'Non-linear' else 'blue'
            style = 'dashed' if edge_type == 'Non-linear' else 'solid'
            rad = 0.1 if edge_type == 'Non-linear' else -0.1

            arrow = FancyArrowPatch(posA=pos[u], posB=pos[v], arrowstyle='-|>', color=color,
                                    linestyle=style, connectionstyle=f'arc3,rad={rad}', linewidth=2, alpha=0.5)

            plt.gca().add_patch(arrow)

            # Calculate label position with offset
            label_x = (pos[u][0] + pos[v][0]) / 2
            label_y = (pos[u][1] + pos[v][1]) / 2 + (0.1 if edge_type == 'Non-linear' else -0.1)

            plt.text(label_x, label_y, f'{weight:.2f}', color='black', fontsize=10,
                     ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    draw_edges_with_arrows_and_labels()

    plt.axis('off')
    plt.title("Network of Components Linked by Correlation Strength")
    plt.show()



def plot_component_comparison(component1, component2, linear_corr_dict_1, non_linear_corr_dict_1, linear_corr_dict_2, non_linear_corr_dict_2, lower_threshold=0.5, upper_threshold=1, top_n_corrs=0, layout=nx.spring_layout):
    """
    Plot a network of two components linked by correlation strength, showing an edge if either linear or non-linear
    correlation is above the minimum threshold. The seed nodes are highlighted in yellow and orange.
    """
    G = nx.MultiDiGraph()

    # Ensure top_n_corrs is an integer
    if not isinstance(top_n_corrs, int):
        raise ValueError("top_n_corrs must be an integer.")

    # Add the first component and its correlations
    G.add_node(component1.name, node_color='yellow')
    if top_n_corrs > 0:
        # Add top N correlations for the first component
        for component, correlation in sorted(linear_corr_dict_1.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n_corrs]:
            if component != component1.name:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Linear')

        for component, correlation in sorted(non_linear_corr_dict_1.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n_corrs]:
            if component != component1.name:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Non-linear')

    else:
        # Add correlations within thresholds for the first component
        for component, correlation in linear_corr_dict_1.items():
            if lower_threshold <= abs(correlation) <= upper_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Linear')

        for component, correlation in non_linear_corr_dict_1.items():
            if lower_threshold <= abs(correlation) <= upper_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component1.name, component, weight=correlation, correlation_type='Non-linear')

    # Add the second component and its correlations
    G.add_node(component2.name, node_color='orange')
    if top_n_corrs > 0:
        # Add top N correlations for the second component
        for component, correlation in sorted(linear_corr_dict_2.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n_corrs]:
            if component != component2.name:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component2.name, component, weight=correlation, correlation_type='Linear')

        for component, correlation in sorted(non_linear_corr_dict_2.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n_corrs]:
            if component != component2.name:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component2.name, component, weight=correlation, correlation_type='Non-linear')

    else:
        # Add correlations within thresholds for the second component
        for component, correlation in linear_corr_dict_2.items():
            if lower_threshold <= abs(correlation) <= upper_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component2.name, component, weight=correlation, correlation_type='Linear')

        for component, correlation in non_linear_corr_dict_2.items():
            if lower_threshold <= abs(correlation) <= upper_threshold:
                G.add_node(component, correlation_strength=correlation)
                G.add_edge(component2.name, component, weight=correlation, correlation_type='Non-linear')

    plt.figure(figsize=(12, 12))
    pos = layout(G)

    # Draw nodes and labels
    node_colors = ['yellow' if node == component1.name else 'orange' if node == component2.name else 'skyblue' for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # Draw edges with straight lines and labels
    edge_colors = ['blue' if attribs['correlation_type'] == 'Linear' else 'red' for _, _, attribs in G.edges(data=True)]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, alpha=0.5)

    # Add edge labels
    edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Create a legend
    legend_elements = [
        Patch(facecolor='yellow', edgecolor='black', label=f'{component1.name} Node'),
        Patch(facecolor='orange', edgecolor='black', label=f'{component2.name} Node'),
        Line2D([0], [0], color='blue', lw=2, label='Linear Correlation'),
        Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Non-linear Correlation')
    ]
    plt.legend(handles=legend_elements, loc='best')



    plt.axis('off')
    plt.title(f"Network of {component1.name} and {component2.name} Strongest Correlations")

    #plt.savefig('D:\GitHub\MSc-Project\Write_Up\Charts\lit_301_401_corr_comparison.jpeg', format='jpeg')

    plt.show()