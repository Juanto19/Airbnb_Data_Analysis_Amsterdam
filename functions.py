import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px

from scipy import stats

from jupyter_dash import JupyterDash
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
#Paleta de colores

palette = ['#00bcFF', '#ff9b00', '#06ae1f', '#ef57b3', '#c8cf00', '#0e4fc8', '#22cf81', '#ac1cde', '#a17e17', '#e70b00']


#Functions to create graphs for univariate analysis with:

#Cathegorical variables:
def cathegorical_simple(df, variables, color = '#00bcFF'):
    """
    Generates count plots for categorical variables.

    This function creates a grid of count plots for the given categorical variables (`variables`) from the dataframe (`df`).
    Each plot shows the distribution of the categories within the variable, with the percentage of each category displayed above the bars.
    The plots are displayed in a grid layout.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - variables (list): A list of categorical variables to be plotted.
    - palette (list): A list of colors to be used for the bars in the plots. Default is the provided `palette`.

    Returns:
    None: Displays the count plots with percentages.
    """
    # Get the dimensions of the grid
    num_rows = len(variables) // 2
    if len(variables) % 2 == 1:
        num_rows += 1

    # Create the grid
    plt.figure(figsize=(12, num_rows * 4))

    # Iterate over the variables
    for i, var in enumerate(variables):
        plt.subplot(num_rows, 2, i + 1)

        # Take the order
        category_order = df[var].value_counts().index
        # Plot the countplot
        plot = sns.countplot(data=df, x=var, color=color, edgecolor='black', order=category_order, legend=True)

        # Calculate percentage using the actual height of the bar, so we don't have to calculate them previously
        total_count = len(df[var])
        slots = df[var].nunique()
        for p in plot.patches:
            height = p.get_height()
            if height > 0:  # Only display percentage if height is greater than zero
                percentage = (height / total_count) * 100
                # Show the % over the bar
                width = p.get_width()
                plt.text(p.get_x() + width / 2., height + (total_count*(1/300)), f'{height} ({percentage:.1f}%)', ha="center", fontsize=11-(slots*0.45))

        #Remove top and right spines
        sns.despine()

        # Add title and labels
        plt.title(f'Countplot of variable {var}')
        if variables.index(var) % 2 == 0:
            plt.ylabel('Count')
        else:
            plt.ylabel('')

    plt.tight_layout()
    plt.show()



#Numerical variables
def numerical_simple(df, variables, color='#00bcFF'):
    """
    Generates histograms and boxplots for numerical variables.

    This function creates a grid of plots for the given numerical variables (`variables`) from the dataframe (`df`).
    For each variable, a histogram with a kernel density estimate (KDE) and a boxplot are generated.
    The plots are displayed in a grid layout with histograms on the top row and boxplots on the bottom row.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - variables (list): A list of numerical variables to be plotted.
    - color (str): The color to be used for the plots. Default is '#00bcFF'.

    Returns:
    None: Displays the histograms and boxplots.
    """
    # Create the grid
    fig, axis = plt.subplots(2, len(variables), figsize=(16, 8), gridspec_kw={'height_ratios': [5, 1]})

    # If there are multiple variables
    if len(variables) > 1:
        # Iterate over the variables
        for var in variables:
            # Enable grid for y-axis
            axis[0, variables.index(var)].grid(visible=True, axis='y', zorder=0)

            # Plot histogram and boxplot
            sns.histplot(ax=axis[0, variables.index(var)], data=df, x=var, kde=True, color=color, zorder=3)
            sns.boxplot(ax=axis[1, variables.index(var)], data=df, x=var, color=color)

            # Remove spines
            sns.despine(ax= axis[0, variables.index(var)], top=True, right=True, left=False, bottom=False)
            sns.despine(ax= axis[1, variables.index(var)], top=True, right=True, left=False, bottom=False)
            
            # Set titles
            axis[0, variables.index(var)].set_title(f'Histograma de {var}')
            axis[1, variables.index(var)].set_title(f'Boxplot de {var}')
    else:
        var = variables[0]
        # Enable grid for y-axis
        axis[0].grid(visible=True, axis='y', zorder=0)

        # Plot histogram and boxplot
        sns.histplot(ax=axis[0], data=df, x=var, kde=True, color=color, zorder=3)
        sns.boxplot(ax=axis[1], data=df, x=var, color=color)

        # Remove spines
        sns.despine(ax= axis[0], top=True, right=True, left=False, bottom=False)
        sns.despine(ax= axis[1], top=True, right=True, left=False, bottom=False)

        # Set titles
        axis[0].set_title(f'Histograma de {var}')
        axis[1].set_title(f'Boxplot de {var}')

    # Adjust layout
    plt.tight_layout()
    # Display plots
    plt.show()


def px_violin_simple(df, y):
    """
    Generates a violin plot for the distribution of a specified column.

    This function creates a violin plot for the given numerical variable (`y`) from the dataframe (`df`).
    The plot includes a box plot and individual points within the violins.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - y (str): The numerical variable to be represented on the y-axis.

    Returns:
    None: Displays the violin plot.
    """
    fig = px.violin(df, y=y, box=True, points="all", hover_data=df.columns)
    fig.update_layout(width=800, height=600)  # Adjust the width and height as needed
    fig.update_layout(title_text=f'Distribution of {y}', title_x=0.5)
    fig.show()



#Functions to create graphs for bivariate analysis with:

#Cathegorical variables:

#       I define a function to create a series of graphs representing the distribution of different variables in regards to another given variable
def cathegorical_pairs(df, dep_var, ind_vars, palette=palette):
    """
    Generates count plots for categorical variables with respect to a dependent variable.

    This function creates a series of count plots for the given independent categorical variables (`ind_vars`).
    Each plot shows the distribution of the dependent variable (`dep_var`) within each category of the independent variable.
    The plots are displayed in a grid layout.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - dep_var (str): The dependent variable to be represented in the count plots.
    - ind_vars (list): A list of independent categorical variables to be plotted.
    - palette (list): A list of colors to be used for the different categories of `dep_var`. Default is the global `palette` variable.

    Returns:
    None: Displays the count plots.
    """
    # Reorder the df to have color consistency
    df = df.sort_values(by=dep_var, ascending=True)
    
    # Select the number of rows given the number of variables
    num_rows = len(ind_vars) // 2
    if len(ind_vars) % 2 == 1:
        num_rows += 1

    # Create the grid
    plt.figure(figsize=(12, 4 * num_rows))
    
    # Iterate over the variables
    for i, var in enumerate(ind_vars):
        # Create the subplot
        plt.subplot(num_rows, 2, i + 1)

        # Take the order
        category_order = df[var].value_counts().index

        #set the palette to use
        palette_plot = palette[:len(df[dep_var].unique())]

        # Plot the countplot
        plot = sns.countplot(data=df, x=var, hue=dep_var, palette=palette_plot, edgecolor='black', order=category_order)
        
        total_count = len(df[var])
        slots = df[var].nunique()
        for p in plot.patches:
            height = p.get_height()
            if height > 0:  # Only display percentage if height is greater than zero
                # Show the % over the bar
                width = p.get_width()
                plt.text(p.get_x() + width / 2., height + (total_count*(1/300)), f'{int(height)}', ha="center", fontsize=11-(slots*0.5))

        plt.title(f'Countplot of {dep_var} as a function of {var}')
    
    sns.despine()
    plt.tight_layout()
    plt.show()


#Mixed variables
def mixed_pairs(df, num_var, cath_var, palette=palette):
    """
    Generates density distribution plots for a numerical variable, separated by a categorical variable.

    This function creates a series of density distribution plots for a given numerical variable (`num_var`).
    The plots are separated into different columns based on the categorical variable (`cath_var`).
    Vertical dashed lines are added to indicate the points of maximum density for each distribution.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - num_var (str): The numerical variable to be represented in the density plots.
    - cath_var (str): The categorical variable used to separate the data into different columns.
    - palette (list): A list of colors to be used for the different categories of `cath_var`. Default is the provided `palette`.

    Returns:
    None: Displays the density distribution plots.
    """
    palette_plot = palette[:len(df[cath_var].unique())]
    # Create the plot
    g = sns.displot(data=df, x=num_var, kde=True, col=cath_var, hue=cath_var, palette=palette_plot, zorder=3, bins=25)

    # Apply the grid to every subplot
    for ax in g.axes.flat:
        ax.grid(True, axis='y', zorder=0)  

    # Apply a main title
    g.figure.suptitle(f'Density distribution of {num_var} according to {cath_var}', fontsize=16)
    plt.subplots_adjust(top=0.85, hspace=0.4)
    plt.show()


def violin_plot_pairs(df, num_var, cath_var, palette=palette):
    """
    Generates a violin plot for a numerical variable, separated by a categorical variable.

    This function creates a violin plot for the given numerical variable (`num_var`) from the dataframe (`df`).
    The plot is separated into different columns based on the categorical variable (`cath_var`).
    Box plots and individual points are also displayed within the violins.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - num_var (str): The numerical variable to be represented on the y-axis.
    - cath_var (str): The categorical variable used to separate the data into different columns.
    - palette (list): A list of colors to be used for the different categories of `cath_var`. Default is the provided `palette`.

    Returns:
    None: Displays the violin plot.
    """
    palette_plot = palette[:len(df[cath_var].unique())]
    sns.violinplot(data=df, x=cath_var, y=num_var, palette=palette_plot, inner="box", scale="width")
    sns.stripplot(data=df, x=cath_var, y=num_var, color='k', alpha=0.5, jitter=True)

    plt.title(f'Violin plot of {num_var} by {cath_var}')
    plt.show()



#Numerical_variables

def numerical_pairs(df, x_var, y_var, color = '#00bcFF'):
    """
    Generates a scatter plot with a regression line for two numerical variables.

    This function creates a scatter plot for the given numerical variables `x_var` and `y_var` from the dataframe `df`.
    A regression line is added to the plot to show the relationship between the two variables.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
     -x_var (str): The name of the column to be used as the x-axis variable.
     -y_var (str): The name of the column to be used as the y-axis variable.
     -color (str): The color to be used for the scatter points. Default is '#00bcFF'.

    Returns:
    None: Displays the scatter plot with a regression line.
    """
    #calculate linear regression and R squared value 
    slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_var], df[y_var])
    r_squared = r_value**2
    
    #Plot the data
    sns.lmplot(data= df, x=x_var, y = y_var, height=4, aspect=4, scatter_kws={'color': color}, line_kws={'color': 'red'})

    #Represent linear regression equation and R squared value 
    equation_text = f"y = {slope:.2f}x + {intercept:.2f}"
    r_squared_text = f"R² = {r_value**2:.3f}"

    height = df[y_var].max()
    width = df[x_var].min()
    plt.text(x=width, y=(0.8*height), s=equation_text, fontsize=12)
    plt.text(x=width+(0.02*width), y=(0.75*height)-15, s=r_squared_text, fontsize=12)

    # plt.grid(True)
    plt.title(f'Scatterplot of {x_var} vs {y_var}')
    plt.show()




#Functions to create graphs for multivariate analysis with:


#mixed variables
def mixed_trios(df, num_var, cath_inter, cath_intra, palette = palette):
    """
    Generates density distribution plots for a numerical variable, separated by two categorical variables.

    This function creates a series of density distribution plots for a given numerical variable (`num_var`).
    The plots are separated into different columns based on the first categorical variable (`cath_inter`),
    and within each column, the density distributions are further separated by the second categorical variable (`cath_intra`).
    Vertical dashed lines are added to indicate the points of maximum density for each distribution.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
     -num_var (str): The numerical variable to be represented in the density plots.
     -cath_inter (str): The categorical variable used to separate the data into different columns.
     -cath_intra (str): The categorical variable used to separate the density distributions within each column.
     -palette (list): A list of colors to be used for the different categories of `cath_intra`. Default is the provided `palette`.

    Returns:
    None: Displays the density distribution plots.
    """

    palette_plot = palette[:len(df[cath_intra].unique())]
    g = sns.displot(data=df, x=num_var, kind='kde', col=cath_inter, hue=cath_intra, palette=palette_plot)

    for ax in g.axes.flat:
        # Get the density line info from the graph
        for line in ax.get_lines():
            data = line.get_data()
            x_values = data[0]
            y_values = data[1]

            # Find the max value
            x_max = x_values[y_values.argmax()]

            # Get the color of the line
            line_color = line.get_color()

            # Draw the vertical line at max density
            ax.axvline(x=x_max, color=line_color, linestyle='--', alpha=0.7, label=f'Max Density at {x_max:.2f}')
            
        ax.legend()

    g.figure.suptitle(f'Density distribution of {num_var} according to {cath_inter} and {cath_intra}', fontsize=16)
    plt.subplots_adjust(top=0.85, hspace=0.4)

    plt.show()


def px_violin_multiple(dataframe, y, x, color):
    """
    Generates a violin plot for a numerical variable, separated by a categorical variable and colored by another categorical variable.

    This function creates a violin plot for the given numerical variable (`y`) from the dataframe (`dataframe`).
    The plot is separated into different columns based on the categorical variable (`x`) and colored by another categorical variable (`color`).
    Box plots and individual points are also displayed within the violins.

    Parameters:
    - dataframe (pandas.DataFrame): The input dataframe containing the data.
    - y (str): The numerical variable to be represented on the y-axis.
    - x (str): The categorical variable used to separate the data into different columns.
    - color (str): The categorical variable used to color the violins.

    Returns:
    None: Displays the violin plot.
    """
    fig = px.violin(dataframe, y=y, x=x, color=color, box=True, points="all", hover_data=dataframe.columns, template="plotly_dark")
    fig.update_layout(title_text='Violin plot of Age vs Survived colored by Sex', title_x=0.5)
    fig.show()


# Correlation matrix
def corr_matrix(df, palette='coolwarm', corr_columns=["Age", "Fare", "Sex_male", "Sex_female", "Pclass", 
                                "Embarked_S", "Embarked_Q", "Embarked_C", 'n_fam', 'alone', 'has_deck', "Survived"]):
    """
    Generates and displays a correlation matrix heatmap for the specified columns in the dataframe.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the data.
    - colors (str): The colormap to be used for the heatmap. Default is 'inferno'.
    - encoding_cols (list): List of categorical columns to be one-hot encoded. Default is ['Sex', 'Embarked'].
    - corr_columns (list): List of columns for which the correlation matrix is to be computed. Default includes 
                         ["Age", "Fare", "Sex_male", "Sex_female", "Pclass", "Embarked_S", "Embarked_Q", "Embarked_C", 
                         'n_fam', 'alone', 'has_deck', "Survived"].

    Returns:
    None: Displays a heatmap of the correlation matrix.
    """
    # Create correlation matrix
    correlation_matrix = df[corr_columns].corr()

    #Represent the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap=palette, vmin=-1, vmax=1)

    plt.title("Correlation Matrix")
    plt.show()

