# module for component code
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import FancyArrowPatch, Patch, ArrowStyle
from matplotlib.lines import Line2D
from matplotlib.text import Annotation

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# for KDE
from scipy.stats import gaussian_kde

# for parallel processing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# for serialising object
import pickle

# class to act as knowledge object for each component in the system
class Component:
    def __init__(self, name, list_position,  df_orig, min_value=None, max_value=None, step_size=None, correlation_dict=None, p_value_dict=None,
                 non_lin_correlation_dict=None, non_lin_p_value_dict=None, mutual_info_dict=None, wave_diff_dict=None   ):
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size

        # position in results list
        self.list_position = list_position
        self.top_n_corrs = 0

        # empty dict for results from EDA for components relationship to all other components
        self.correlation_dict = correlation_dict if correlation_dict is not None else {}
        self.p_value_dict = p_value_dict if p_value_dict is not None else {}
        self.non_lin_correlation_dict = non_lin_correlation_dict if non_lin_correlation_dict is not None else {}
        self.non_lin_p_value_dict = non_lin_p_value_dict if p_value_dict is not None else {}
        self.mutual_info_dict = mutual_info_dict if mutual_info_dict is not None else {}
        self.wave_diff_dict = wave_diff_dict if wave_diff_dict is not None else {}

        # Initialise min, max, and step size
        self.min_value, self.max_value, self.step_size = self.calculate_min_max_step_size(df_orig)


        # Initialie wave difference dict
        #self.wave_diff_dict = wave_diff_dict if wave_diff_dict is not None else self.calculate_wave_differences(df_orig)


    def calculate_min_max_step_size(self, df):
        """
        Calculates the min, max and largest step size for the component.
        """
        if self.name in df.columns:
            column_data = df[self.name].dropna()
            min_value = column_data.min()
            max_value = column_data.max()

            # Compute step size as the largest difference between consecutive time steps
            step_size = column_data.diff().abs().max() if len(column_data) > 1 else 0

            return min_value, max_value, step_size
        else:
            # return default values
            return 0, 0, 0


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

    # return n strongerst correlations
    def get_mutual_info(self, top_mi_components = -1):
        """
        Returns mutual information, takes optional number or strongest mi

        Parameters:
        - top_mi_components: Number of strongest mi to return, default is all

        Returns:
        - A dictionary containing component names as keys and correlation values as values.
        """

        filtered_dict = {k: v for k, v in self.mutual_info_dict.items() if np.isfinite(v)}

        sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True))

        return dict(list(sorted_dict.items())[:top_mi_components])

    def calculate_wave_differences(self, df):
        """
        Calculate phase and time differences between this component and all other components.
        """
        wave_diff_dict = {}

        # Loop through all other columns in the DataFrame
        for column in df.columns:
            if column != self.name:
                phase_diff, time_diff = calculate_phase_and_time_difference(df, self.name, column)
                wave_diff_dict[column] = {'phase_diff': phase_diff, 'time_diff': time_diff}

        return wave_diff_dict
    def calculate_phase_and_time_difference(df_orig, column1, column2, dt=1):
        """
        Calculate the phase difference and time difference between the fundamental frequencies of two signals.
        """



        # Perform Fourier analysis on both columns
        data1 = df[column1].values
        data2 = df[column2].values

        fft_data1 = np.fft.fft(data1)
        fft_data2 = np.fft.fft(data2)

        freq1 = np.fft.fftfreq(len(data1), dt)
        freq2 = np.fft.fftfreq(len(data2), dt)

        # Identify fundamental frequencies
        power_spectrum1 = np.abs(fft_data1) ** 2
        power_spectrum2 = np.abs(fft_data2) ** 2

        peak_indices1, _ = find_peaks(power_spectrum1)
        peak_indices2, _ = find_peaks(power_spectrum2)

        # Check if peaks were found
        if len(peak_indices1) == 0 or len(peak_indices2) == 0:
            # return None or zeros)
            return None, None

        peak_freq1 = freq1[peak_indices1[np.argmax(power_spectrum1[peak_indices1])]]
        peak_freq2 = freq2[peak_indices2[np.argmax(power_spectrum2[peak_indices2])]]

        # Find the corresponding phase at the fundamental frequencies
        phase1 = np.angle(fft_data1[peak_indices1[np.argmax(power_spectrum1[peak_indices1])]])
        phase2 = np.angle(fft_data2[peak_indices2[np.argmax(power_spectrum2[peak_indices2])]])

        # Calculate the phase difference
        phase_difference = phase2 - phase1

        # Calculate the time difference
        avg_freq = (peak_freq1 + peak_freq2) / 2  # Average fundamental frequency
        time_difference = phase_difference / (2 * np.pi * avg_freq)


        # default values
        phase_difference= 9999
        time_difference= 9999

        return phase_difference, time_difference



def save_component(component_obj, filename):
    ''' Function to save the Component object, use .pkl extension'''
    with open(filename, 'wb') as file:
        pickle.dump(component_obj, file)
    print(f"Component '{component_obj.name}' saved to {filename}.")


def load_component(filename):
    ''' Function to load the Component object'''
    with open(filename, 'rb') as file:
        component_obj = pickle.load(file)
    print(f"Component '{component_obj.name}' has been loaded from {filename}.")
    return component_obj




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

def drop_static_columns(df: pd.DataFrame) -> (pd.DataFrame, list[str]):
    """
    Find columns in the DataFrame where all values are the same (static columns).

    Parameters:
        df : All values from SWaT .

    Returns:
        df_cleaned: Columns from SWaT which have more than 1 value.
        static_columns: dropped columns

    """
    column_names = df.columns
    static_columns = []
    for column in column_names:
        if df[column].min() == df[column].max():
            static_columns.append(column)

    df_cleaned = df.drop(columns= static_columns)
    return df_cleaned, static_columns

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



def getDistributionBins(df, variable_col, save_path = None):
    '''
    Function to get distribution of values in variable column and plots as binned values

    Parameters:
        df: SWaT Data
        variable_col: pandas series with component values
        save_path: optional if jpeg of plot needed
    Returns
    '''


    # df to hold variable distributions
    df_distributions = pd.DataFrame()
    #get variable min and max values and calculate range



    var_range = df[variable_col].max() - df[variable_col].min()

    print(f'Var Range:  {var_range}')

    # divide into 100 bins
    bin_size = var_range/100

    # Create bins edges from min to max with using bin_size'
    bins = np.arange(df[variable_col].min(), df[variable_col].max() + bin_size, bin_size)

    # Use pd.cut to segment and sort the data values into bins
    df_distributions[variable_col + '_bins'] = pd.cut(df[variable_col], bins=bins, include_lowest=True)

    # Count the number of values in each bin
    bin_counts = df_distributions[variable_col + '_bins'].value_counts().sort_index()

    # call sisualisation function
    plotBinCounts(variable_col, bin_counts, bins, save_path)

    # return bin_counts, bins


def plotBinCounts(name, bin_counts, bin_edges, save_path):
    '''
    Function to plot the distribution of bin counts.
    Parameters:
    - bin_counts: a pandas Series containing the counts of values in each bin.
    - bin_edges: an array containing the bin edges.
    '''

    # Set the size of the plot
    plt.figure(figsize=(10,6))

    # Plot the bin counts
    bin_counts.plot(kind='bar', logy=True)

    # Set the title and labels
    plt.title(f'Distribution of {name}')
    plt.ylabel('Counts')

    # Calculate total range and step size for approximately 10 divisions
    total_range = bin_edges[-1] - bin_edges[0]
    step_size = total_range / 10

    # Round step size to a whole number (e.g., 10, 20, etc.) that makes sense for your data
    rounded_step_size = round(step_size / 10) * 10

    # Generate custom tick positions and labels
    tick_positions = np.arange(0, len(bin_edges) - 1, rounded_step_size / (bin_edges[1] - bin_edges[0]))
    tick_labels = [f"{bin_edges[int(pos)]:.0f}" for pos in tick_positions]

    # Set x-ticks to represent the overall value range of the bins
    plt.xticks(ticks=tick_positions,
               labels=tick_labels,
               rotation=45)  # Rotate labels for better readability

    # Optional: Set x-axis label
    plt.xlabel('Value Range')

    if save_path:
        plt.savefig(save_path, format='jpeg')

    # Show the plot
    plt.show()


def getKDEDensity(df, variable_col, bandwidth_adjust=1, save_path=None):
    '''
    Generate and plot kernel density estimate for a variable column using gaussian_kde.

    Parameters:
    df: SWaT Data
    variable_col: component name in df
    bandwidth_adjust: factor to adjust bandwidth/ STD of the KDE
    save_path: Path to save image to, format is jpeg

    Returns:
    Nothing but plots KDE and save image is path is given.
    '''
    # Check if there are any missing values in the variable column
    if df[variable_col].isnull().sum() > 0:
        print(f"Warning: {df[variable_col].isnull().sum()} missing values in {variable_col}, filling with median.")
        df[variable_col].fillna(df[variable_col].median(), inplace=True)

    # Extract the data and convert to numpy array
    data = df[variable_col].dropna().values

    # Calculate KDE using scipy's gaussian_kde
    kde = gaussian_kde(data, bw_method='scott' * bandwidth_adjust)

    # Create a grid of values over which to evaluate the KDE
    x_grid = np.linspace(data.min(), data.max(), 1000)

    # Evaluate KDE on the grid
    kde_values = kde(x_grid)

    # Plot the KDE
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, kde_values, color='blue', lw=2)
    plt.fill_between(x_grid, kde_values, color='skyblue', alpha=0.5)

    # Add title and labels
    plt.title(f'Kernel Density Estimate of {variable_col}', fontsize=16)
    plt.xlabel(variable_col, fontsize=12)
    plt.ylabel('Density', fontsize=12)

    if save_path:
        plt.savefig(save_path, format='jpeg')

    # Show plot
    plt.show()


def compute_mutual_information(x_col, y_col, num_points=100):
    '''
    Helper function to compute mutual information between two columns using KDE.
    '''
    # Kernel Density Estimate for joint distribution p(x, y)
    kde_joint = gaussian_kde(np.vstack([x_col, y_col]))

    # Kernel Density Estimate for marginal distributions p(x) and p(y)
    kde_x = gaussian_kde(x_col)
    kde_y = gaussian_kde(y_col)

    # Create a grid of values where we will estimate the densities
    x_min, x_max = x_col.min(), x_col.max()
    y_min, y_max = y_col.min(), y_col.max()

    # Create grid points
    x_grid = np.linspace(x_min, x_max, num_points)
    y_grid = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])

    # Estimate densities on the grid
    p_xy = kde_joint(positions).reshape(num_points, num_points)  # Joint density
    p_x = kde_x(x_grid)  # Marginal density for x
    p_y = kde_y(y_grid)  # Marginal density for y

    # Normalize the densities to get probability values
    p_xy /= np.sum(p_xy)  # Normalize joint density
    p_x /= np.sum(p_x)    # Normalize x marginal density
    p_y /= np.sum(p_y)    # Normalize y marginal density

    # Calculate mutual information
    mi = 0.0
    for i in range(num_points):
        for j in range(num_points):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j]))

    return mi

def kde_mutual_information(df, component_name, num_points=100, n_jobs=-1):
    '''
    Calculate mutual information between a given component and all other components using KDE.

    Parameters:
    df: pd.DataFrame - Dataset
    component_name: str - Component/column name for which to calculate MI against others
    num_points: int - Number of grid points for KDE (default: 100)
    n_jobs: int - Number of parallel jobs (default: -1, use all available cores)

    Returns:
    pd.Series - Mutual information between the component and all other columns in the DataFrame.
    '''

    # Select the component/column of interest
    x_col = np.array(df[component_name])

    # Define a helper function for parallel execution
    def compute_mi_for_column(other_component):
        if other_component == component_name:
            return None  # Skip if it's the same component
        y_col = np.array(df[other_component])
        mi = compute_mutual_information(x_col, y_col, num_points)
        return other_component, mi

    # Use joblib to parallelize the computation
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_mi_for_column)(col) for col in df.columns if col != component_name
    )

    # Convert the results to a dictionary and then to a pandas Series
    mutual_info_results = {key: value for key, value in results if key is not None}

    # Return as a pandas Series for easy access
    return pd.Series(mutual_info_results)




def doFourierAnalysis(df, column_name, dt=1):

    """
    Perform Fourier analysis on a specified column of a DataFrame and return the frequency bins and power spectrum.

    This function computes the Fast Fourier Transform (FFT) of the data in the specified column,
    calculates the power spectrum, and returns the power spectrum along with the frequency bins.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to analyze.
    column_name (str): The name of the column in the DataFrame to analyze.
    dt (float): The time step between data points.

    Returns:
    tuple: A tuple containing the frequency bins and the power spectrum.
    """
    # Extract the data from the DataFrame
    data = df[column_name].values

    # Number of data points
    N = len(data)

    # Compute the Fast Fourier Transform (FFT)
    fft_data = np.fft.fft(data)

    # Compute the Power Spectrum (magnitude of the FFT squared)
    power_spectrum = np.abs(fft_data) ** 2

    # Compute the frequency bins
    freq = np.fft.fftfreq(N, dt)

    # Remove the DC component (zero frequency)
    # Find indices where the frequency is greater than 0
    valid_indices = freq > 0

    freq = freq[valid_indices]
    power_spectrum = power_spectrum[valid_indices]

    return freq, power_spectrum


def get_harmonic_time_differences(df, column_name, dt=1):

    """
    Calculate the time differences between peak values of the signal for the harmonics.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to analyze.
    column_name (str): The name of the column in the DataFrame to analyze.
    dt (float): The time step between data points.

    Returns:
    dict: A dictionary containing the harmonic frequencies and their corresponding time differences.
    """
    # Perform Fourier analysis
    freq, power_spectrum = doFourierAnalysis(df, column_name, dt)

    # Find the peak frequency (fundamental frequency)
    peak_indices, _ = find_peaks(power_spectrum)
    peak_freq = freq[peak_indices[np.argmax(power_spectrum[peak_indices])]]
    fundamental_period = 1 / peak_freq

    # Identify harmonics
    harmonics = [peak_freq * n for n in range(2, 6)]  # First four harmonics

    # Calculate time differences
    time_differences = {f"Harmonic {n} ({harm}) Hz": fundamental_period / n for n, harm in enumerate(harmonics, start=2)}

    return time_differences

def performFourierAndLimitHarmonics(df=None, column_name=None,  series=None, num_harmonics=3):

    if df is not None and column_name is not None:
        data = df[column_name].values
    elif series is not None and isinstance(series, pd.Series):
        data = series.values

    fft_data = np.fft.fft(data)

        
    N = len(data)

    # Compute the magnitude of the FFT and find the indices of the largest components
    magnitudes = np.abs(fft_data)
    indices = np.argsort(magnitudes)[::-1]  # Sort indices by magnitude in descending order

    # Zero out all but the largest `num_harmonics` components
    fft_data_limited = np.zeros(N, dtype=complex)
    for i in range(num_harmonics):
        index = indices[i]
        fft_data_limited[index] = fft_data[index]

    # Inverse FFT to reconstruct the signal with limited harmonics
    reconstructed_signal = np.fft.ifft(fft_data_limited)

    return reconstructed_signal, fft_data_limited




# %%
#  Function to find the fundamental frequency
def find_fundamental_frequency(freq, power_spectrum):
    # Peaks in the power spectrum
    peaks, _ = find_peaks(power_spectrum)

    # Fundamental frequency (the first peak)
    if peaks.size > 0:
        fundamental_freq = freq[peaks[0]]  # The first peak corresponds to the fundamental frequency
        return fundamental_freq
    else:
        return None

def calculate_phase_and_time_difference(df, column1, column2, dt=1):
    """
    Calculate the phase difference and time difference between the fundamental frequencies of two signals.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to analyze.
    column1 (str): The name of the first column in the DataFrame.
    column2 (str): The name of the second column in the DataFrame.
    dt (float): The time step between data points.

    Returns:
    tuple: A tuple containing the phase difference in radians and the time difference in seconds.
    """
    # Perform Fourier analysis on both columns
    data1 = df[column1].values
    data2 = df[column2].values

    fft_data1 = np.fft.fft(data1)
    fft_data2 = np.fft.fft(data2)

    freq1 = np.fft.fftfreq(len(data1), dt)
    freq2 = np.fft.fftfreq(len(data2), dt)

    # Identify fundamental frequencies
    power_spectrum1 = np.abs(fft_data1) ** 2
    power_spectrum2 = np.abs(fft_data2) ** 2

    peak_indices1, _ = find_peaks(power_spectrum1)
    peak_indices2, _ = find_peaks(power_spectrum2)

    peak_freq1 = freq1[peak_indices1[np.argmax(power_spectrum1[peak_indices1])]]
    peak_freq2 = freq2[peak_indices2[np.argmax(power_spectrum2[peak_indices2])]]

    # Find the corresponding phase at the fundamental frequencies
    phase1 = np.angle(fft_data1[peak_indices1[np.argmax(power_spectrum1[peak_indices1])]])
    phase2 = np.angle(fft_data2[peak_indices2[np.argmax(power_spectrum2[peak_indices2])]])

    # Calculate the phase difference
    phase_difference = phase2 - phase1

    # Calculate the time difference
    avg_freq = (peak_freq1 + peak_freq2) / 2  # Average fundamental frequency
    time_difference = phase_difference / (2 * np.pi * avg_freq)

    return phase_difference, time_difference


def scaler_sec_midnight(csv_path, converted_path, scaler_type='original'):
    '''
    Converts csv to one with seconds since midnight as the index and min max scaled so values are between o and 1
    :param csv_path:
    :param converted_path:
    :para scaler_type: Type of scaler to apply
                        'standard'
                        'minmax'
                        'original'
    :return: converted df
    '''
    df = pd.read_csv(csv_path)

    # df.head()
    df, dropped_cols = drop_static_columns(df)

    # check not already converted to second
    if 'Timestamp' in df.columns:
        # convert time to seconds since midnight
        # Convert 'Timestamp' column to datetime format
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['time_in_seconds'] = (
                    df['Timestamp'].dt.hour * 3600 + df['Timestamp'].dt.minute * 60 + df['Timestamp'].dt.second)
        df.drop(columns=['Timestamp', 'time'], inplace=True)


    df.set_index('time_in_seconds', inplace=True)
    # normalised data
    if scaler_type == 'standard':
        scaler = StandardScaler()
        df_normalised = scaler.fit_transform(df)

    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        df_normalised = scaler.fit_transform(df)

    else:
        df_normalised = df

    # convert back to df with index
    if scaler_type in ['standard', 'minmax']:
        df_normalised = pd.DataFrame(df_normalised, columns=df.columns, index=df.index)

    df_normalised.to_csv(converted_path)

    return df_normalised