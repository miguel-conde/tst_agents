import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Optional

import pandas as pd
import numpy as np


def dataframe_report(data: pd.DataFrame) -> str:
    """
    Generates a detailed Markdown report of a DataFrame's contents including statistics,
    missing values, outliers, and more.

    Parameters:
        data (pd.DataFrame): The DataFrame to analyze.

    Returns:
        str: A Markdown-formatted summary of the DataFrame's contents.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'A': np.random.randn(100),
        ...     'B': np.random.randint(1, 10, 100),
        ...     'C': ['cat', 'dog', 'mouse'] * 33 + ['cat']
        ... })
        >>> print(dataframe_report_markdown(data))
    """
    report = []

    # General overview
    report.append("# General Information")
    report.append(f"- **Shape:** {data.shape}")
    report.append(f"- **Columns:** {', '.join(data.columns)}")
    report.append("\n### Data Types\n")
    report.append(data.dtypes.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}).to_markdown(index=False))
    report.append("")

    # Missing values
    report.append("# Missing Values")
    missing_values = data.isnull().sum()
    report.append(f"\n### Missing Values per Column\n")
    report.append(missing_values.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}).to_markdown(index=False))
    report.append(f"\n- **Total missing values:** {missing_values.sum()}")
    report.append("")

    # Descriptive statistics
    report.append("# Descriptive Statistics")
    report.append("\n### Summary Statistics\n")
    report.append(data.describe(include="all").reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}).to_markdown(index=False))
    report.append("")

    # Outliers (based on IQR)
    report.append("# Outliers Detection")
    numeric_data = data.select_dtypes(include=[np.number])
    if not numeric_data.empty:
        outliers_report = []
        for col in numeric_data.columns:
            q1 = numeric_data[col].quantile(0.25)
            q3 = numeric_data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = numeric_data[(numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)]
            outliers_report.append(f"- **{col}:** {len(outliers)} outliers detected")
        report.append("\n".join(outliers_report))
    else:
        report.append("- No numeric columns to check for outliers.")
    report.append("")

    # Unique values and distribution
    report.append("# Unique Values and Distribution")
    for col in data.columns:
        unique_count = data[col].nunique()
        if unique_count <= 10:  # For small unique values, show distribution
            value_counts = data[col].value_counts().to_markdown()
            report.append(f"\n### {col} - Unique Values: {unique_count}\n")
            report.append(value_counts)
        else:
            report.append(f"- **{col}:** {unique_count} unique values")
    report.append("")

    # Correlations
    report.append("# Correlations")
    correlations = numeric_data.corr()
    if not correlations.empty:
        report.append("\n### Correlation Matrix\n")
        report.append(correlations.reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}).to_markdown(index=False))
    else:
        report.append("- No numeric columns to calculate correlations.")
    report.append("")

    return "\n".join(report)




def plot_time_series(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    save_path: Optional[str] = None,
    title: str = "Time Series Plot",
    subtitle: Optional[str] = None,
    xlabel: str = "Date",
    ylabel: str = "Values",
    figsize: tuple = (12, 6)
) -> None:
    """
    Plots time series data from a DataFrame using matplotlib.

    Parameters:
        data (pd.DataFrame): The DataFrame containing time series data.
        columns (Optional[List[str]]): List of column names to plot. Defaults to all numeric columns.
        date_col (Optional[str]): Name of the date column if not in the index. Defaults to None.
        save_path (Optional[str]): Path to save the figure. Defaults to None.
        title (str): Title of the plot. Defaults to "Time Series Plot".
        subtitle (Optional[str]): Subtitle of the plot. Defaults to None.
        xlabel (str): Label for the x-axis. Defaults to "Date".
        ylabel (str): Label for the y-axis. Defaults to "Values".
        figsize (tuple): Figure size. Defaults to (12, 6).

    Returns:
        None

    Example:
        >>> import pandas as pd
        >>> import utils_EDA as eda
        >>> data = pd.DataFrame({
        ...     'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        ...     'value1': [10, 15, 14, 18, 20, 25, 30, 28, 27, 26],
        ...     'value2': [5, 7, 6, 8, 10, 12, 14, 13, 13, 12]
        ... })
        >>> plot_time_series(
        ...     data=data,
        ...     columns=['value1', 'value2'],
        ...     date_col='date',
        ...     title="Sample Time Series",
        ...     subtitle="This is a subtitle",
        ...     xlabel="Date",
        ...     ylabel="Values"
        ... )
    """
    # Ensure date is in the index
    if date_col:
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
        data = data.set_index(date_col)
    
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index could not be converted to datetime.") from e

    # Select columns to plot
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    if not columns:
        raise ValueError("No numeric columns available for plotting.")
    
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} are not in the DataFrame.")
    
    # Plot the data
    plt.figure(figsize=figsize)
    for col in columns:
        plt.plot(data.index, data[col], label=col)
    
    # Add title and subtitle
    plt.title(title)
    if subtitle:
        plt.suptitle(subtitle, fontsize=10, y=0.95, color='gray')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
    
    plt.show()

import pandas as pd
import plotly.express as px
from typing import List, Optional


def plot_time_series_plotly(
    data: pd.DataFrame,
    columns: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    title: str = "Time Series Plot",
    subtitle: Optional[str] = None,
    xlabel: str = "Date",
    ylabel: str = "Values"
) -> None:
    """
    Plots interactive time series data from a DataFrame using plotly.

    Parameters:
        data (pd.DataFrame): The DataFrame containing time series data.
        columns (Optional[List[str]]): List of column names to plot. Defaults to all numeric columns.
        date_col (Optional[str]): Name of the date column if not in the index. Defaults to None.
        title (str): Title of the plot. Defaults to "Time Series Plot".
        subtitle (Optional[str]): Subtitle of the plot. Defaults to None.
        xlabel (str): Label for the x-axis. Defaults to "Date".
        ylabel (str): Label for the y-axis. Defaults to "Values".

    Returns:
        None

    Example:
        >>> import pandas as pd
        >>> import utils_EDA as eda
        >>> data = pd.DataFrame({
        ...     'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        ...     'value1': [10, 15, 14, 18, 20, 25, 30, 28, 27, 26]*10,
        ...     'value2': [5, 7, 6, 8, 10, 12, 14, 13, 13, 12]*10
        ... })
        >>> eda.plot_time_series_plotly(
        ...     data=data,
        ...     columns=['value1', 'value2'],
        ...     date_col='date',
        ...     title="Sample Time Series with Plotly",
        ...     subtitle="This is a subtitle",
        ...     xlabel="Date",
        ...     ylabel="Values"
        ... )
    """
    # Ensure date is in the index or use the specified column
    if date_col:
        if date_col not in data.columns:
            raise ValueError(f"Date column '{date_col}' not found in the DataFrame.")
        data = data.set_index(date_col)
    
    if not isinstance(data.index, pd.DatetimeIndex):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception as e:
            raise ValueError("Index could not be converted to datetime.") from e

    # Select columns to plot
    if columns is None:
        columns = data.select_dtypes(include=['number']).columns.tolist()
    
    if not columns:
        raise ValueError("No numeric columns available for plotting.")
    
    missing_cols = [col for col in columns if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} are not in the DataFrame.")
    
    # Melt data for plotly compatibility
    melted_data = data[columns].reset_index().melt(id_vars=[data.index.name], var_name="Series", value_name="Value")

    # Create the plot
    fig = px.line(
        melted_data,
        x=data.index.name,
        y="Value",
        color="Series",
        title=title,
        labels={data.index.name: xlabel, "Value": ylabel}
    )
    
    # Add subtitle if provided
    if subtitle:
        fig.update_layout(
            title={
                "text": f"{title}<br><sub>{subtitle}</sub>",
                "x": 0.5,
                "xanchor": "center"
            }
        )
    
    # Show the plot
    fig.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms


def pairs_panel(
    data: pd.DataFrame,
    figsize: tuple = (10, 10),
    hist_color: str = "#4caf50",
    scatter_color: str = "#4caf50",
    corr_cmap: str = "coolwarm",
    fontsize_corr: int = 14,
    fontsize_labels: int = 10,
    fontsize_ticks: int = 8
) -> None:
    """
    Creates a pairwise panel similar to R's psych::pairs.panel(), 
    showing scatter plots with ellipses, histograms, and correlation coefficients.

    Parameters:
        data (pd.DataFrame): The DataFrame containing numeric data to analyze.
        figsize (tuple): Size of the overall plot. Defaults to (10, 10).
        hist_color (str): Color for histograms. Defaults to "#4caf50".
        scatter_color (str): Color for scatter plots. Defaults to "#4caf50".
        corr_cmap (str): Colormap for correlation text. Defaults to "coolwarm".
        fontsize_corr (int): Font size for correlation text. Defaults to 14.
        fontsize_labels (int): Font size for axis labels on the diagonal. Defaults to 10.
        fontsize_ticks (int): Font size for ticks. Defaults to 8.

    Returns:
        None

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = pd.DataFrame({
        ...     'A': np.random.randn(100),
        ...     'B': np.random.randn(100),
        ...     'C': np.random.randn(100) * 2,
        ...     'D': np.random.randn(100) + 1
        ... })
        >>> pairs_panel(data)
    """
    # Select only numeric columns
    data = data.select_dtypes(include=[np.number])
    variables = data.columns.tolist()  # Ensure all columns are included
    num_vars = len(variables)

    if num_vars < 2:
        raise ValueError("Data must contain at least two numeric variables.")

    # Create subplots with n x n grid
    fig, axes = plt.subplots(
        num_vars, num_vars, figsize=figsize, sharex="col", sharey="row"
    )
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, var1 in enumerate(variables):
        for j, var2 in enumerate(variables):
            ax = axes[i, j]
            if i == j:
                # Histogram on the diagonal
                sns.histplot(data[var1], kde=True, color=hist_color, ax=ax)
                ax.set_ylabel("")
                ax.set_xlabel("")
            elif i > j:
                # Scatter plot with linear fit and confidence ellipse
                sns.scatterplot(x=data[var2], y=data[var1], color=scatter_color, alpha=0.6, ax=ax)
                sns.regplot(x=data[var2], y=data[var1], scatter=False, color="black", ax=ax)
                confidence_ellipse(data[var2], data[var1], ax, edgecolor="black")
                ax.tick_params(axis="x", labelsize=fontsize_ticks)
                ax.tick_params(axis="y", labelsize=fontsize_ticks)
            else:
                # Correlation coefficients above the diagonal
                corr, p_value = pearsonr(data[var1], data[var2])
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                color = plt.cm.get_cmap(corr_cmap)((corr + 1) / 2)  # Normalize -1 to 1
                ax.text(
                    0.5,
                    0.5,
                    f"{corr:.2f}{significance}",
                    fontsize=fontsize_corr,
                    color=color,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")

    # Ensure correct alignment of ticks and axes
    for i in range(num_vars):
        for j in range(num_vars):
            if i != num_vars - 1:
                axes[i, j].xaxis.set_visible(j >= i)
            if j != 0:
                axes[i, j].yaxis.set_visible(i >= j)

    plt.show()


def confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
    """
    Add a confidence ellipse to the plot.

    Parameters:
        x, y: Arrays of data points.
        ax: The axis object to draw the ellipse.
        n_std: Number of standard deviations for the ellipse radius.
        facecolor: Fill color of the ellipse.
        kwargs: Additional arguments for Ellipse.

    Returns:
        None
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse_radius_x = np.sqrt(1 + pearson)
    ellipse_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ellipse_radius_x * 2,
        height=ellipse_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a correlation table for numeric columns in a DataFrame.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the correlation matrix.

    Example:
        >>> data = pd.DataFrame({
        ...     'A': np.random.rand(100),
        ...     'B': np.random.rand(100),
        ...     'C': np.random.rand(100)
        ... })
        >>> corr_matrix = correlation_table(data)
        >>> print(corr_matrix)
    """
    # Select only numeric columns
    numeric_data = data.select_dtypes(include=[np.number])
    # Generate correlation matrix
    return numeric_data.corr()


def correlation_heatmap(corr_matrix: pd.DataFrame, figsize: tuple = (10, 8), cmap: str = "coolwarm") -> None:
    """
    Plot a heatmap for a given correlation matrix.

    Parameters:
        corr_matrix (pd.DataFrame): The correlation matrix to visualize.
        figsize (tuple): The size of the heatmap. Defaults to (10, 8).
        cmap (str): The colormap for the heatmap. Defaults to "coolwarm".

    Returns:
        None

    Example:
        >>> data = pd.DataFrame({
        ...     'A': np.random.rand(100),
        ...     'B': np.random.rand(100),
        ...     'C': np.random.rand(100)
        ... })
        >>> corr_matrix = correlation_table(data)
        >>> correlation_heatmap(corr_matrix)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,  # Display correlation coefficients
        fmt=".2f",   # Format numbers with 2 decimals
        cmap=cmap,
        cbar=True,   # Show color bar
        square=True, # Make cells square
    )
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt


def violin_plots(
    data: pd.DataFrame,
    columns: list = None,
    category: str = None,
    figsize: tuple = (10, 6),
    palette: str = "muted",
    y_reference_lines: list = None
) -> None:
    """
    Generate violin plots for specified numeric columns in a DataFrame, optionally subdivided by a categorical variable.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list): A list of numeric column names to plot. Defaults to all numeric columns.
        category (str): A categorical column to subdivide the violins by. Defaults to None.
        figsize (tuple): The size of the plot. Defaults to (10, 6).
        palette (str): The color palette for the violins. Defaults to "muted".
        y_reference_lines (list): A list of Y values to add as reference lines. Defaults to None.

    Returns:
        None
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    invalid_cols = [col for col in columns if col not in data.columns]
    if invalid_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {invalid_cols}")
    
    if category and category not in data.columns:
        raise ValueError(f"The category column '{category}' is not in the DataFrame.")

    melted_data = data[columns + ([category] if category else [])].melt(
        id_vars=[category] if category else None, var_name="Variable", value_name="Value"
    )

    plt.figure(figsize=figsize)
    sns.violinplot(
        x="Variable",
        y="Value",
        hue=category,
        data=melted_data,
        inner="box",
        palette=palette,
        split=True if category else False
    )

    if y_reference_lines:
        for y in y_reference_lines:
            plt.axhline(y=y, color="red", linestyle="--", alpha=0.7)

    plt.title("Violin Plots of Numeric Variables", fontsize=16)
    plt.xlabel("Variables", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title=category, bbox_to_anchor=(1.05, 1), loc="upper left") if category else None
    plt.show()

def box_plots(
    data: pd.DataFrame,
    columns: list = None,
    category: str = None,
    figsize: tuple = (10, 6),
    palette: str = "muted",
    y_reference_lines: list = None
) -> None:
    """
    Generate boxplots for specified numeric columns in a DataFrame, optionally subdivided by a categorical variable.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        columns (list): A list of numeric column names to plot. Defaults to all numeric columns.
        category (str): A categorical column to subdivide the boxes by. Defaults to None.
        figsize (tuple): The size of the plot. Defaults to (10, 6).
        palette (str): The color palette for the boxes. Defaults to "muted".
        y_reference_lines (list): A list of Y values to add as reference lines. Defaults to None.

    Returns:
        None
    """
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    invalid_cols = [col for col in columns if col not in data.columns]
    if invalid_cols:
        raise ValueError(f"The following columns are not in the DataFrame: {invalid_cols}")
    
    if category and category not in data.columns:
        raise ValueError(f"The category column '{category}' is not in the DataFrame.")

    melted_data = data[columns + ([category] if category else [])].melt(
        id_vars=[category] if category else None, var_name="Variable", value_name="Value"
    )

    plt.figure(figsize=figsize)
    sns.boxplot(
        x="Variable",
        y="Value",
        hue=category,
        data=melted_data,
        palette=palette
    )

    if y_reference_lines:
        for y in y_reference_lines:
            plt.axhline(y=y, color="red", linestyle="--", alpha=0.7)

    plt.title("Boxplots of Numeric Variables", fontsize=16)
    plt.xlabel("Variables", fontsize=12)
    plt.ylabel("Values", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title=category, bbox_to_anchor=(1.05, 1), loc="upper left") if category else None
    plt.show()
