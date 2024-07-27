import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Lê o arquivo CSV
df = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

def estilo_tabela(val):
    return 'text-align: center;'

def plot_histogram(column, bins):
    # Set up the figure with two subplots
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(wspace=0.3)

    # Histogram of absolute frequencies
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(df[column], bins=bins, edgecolor='black', color='skyblue')
    plt.title(f'Histograma de Frequência Absoluta: {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequência Absoluta', fontsize=12)
    plt.grid(axis='y', alpha=0.9)

    # Histogram of relative frequencies
    plt.subplot(1, 2, 2)
    weights = (np.ones_like(df[column]) / len(df[column])) * 100  # Weights to convert counts to percentages
    n, bins, patches = plt.hist(df[column], bins=bins, weights=weights, edgecolor='black', color='coral')
    plt.title(f'Histograma de Frequência Relativa: {column}', fontsize=15)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequência Relativa (%)', fontsize=12)
    plt.grid(axis='y', alpha=0.9)

    # Display the plot
    plt.tight_layout()
    plt.show()

def describe(name_column, bins_lenght):
    # Assuming 'Income' is a numerical column
    min_val = df[name_column].min()
    max_val = df[name_column].max()
    amplitude = math.ceil((max_val - min_val)/bins_lenght)
    max_limits = [min_val + amplitude * i for i in range(1, bins_lenght+1)]
    min_limits = [min_val + amplitude * i for i in range(0, bins_lenght)]
    df[name_column+'_class'] = pd.cut(df[name_column], bins=bins_lenght, labels=range(1, bins_lenght+1))
    class_summary = df[name_column+'_class'].value_counts().sort_index().rename('frequency').to_frame()
    class_summary['lower_limit'] = min_limits
    class_summary['upper_limit'] = max_limits
    class_summary['relative_frequency'] = (class_summary['frequency'] / class_summary['frequency'].sum())*100
    class_summary['cumulative_frequency'] = class_summary['frequency'].cumsum()
    class_summary['cumulative_frequency_percentage'] = (class_summary['cumulative_frequency'] / class_summary['frequency'].sum())*100
    styled_df = class_summary.style.apply(estilo_tabela)
    print(styled_df)
    plot_histogram(name_column + '_class', bins_lenght)

describe('Income', 10)