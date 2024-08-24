import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Faz a leitura do arquivo
    input_file = 'C:/MineracaoDados/adult/adultClear.data'
    names = ['Age', 'Workclass', 'Education', 'Education-Num', 'Marital-Status',
             'Occupation', 'Relationship', 'Race', 'Sex', 'Capital-Gain',
             'Capital-Loss', 'Hours-per-week', 'Native-Country', 'Income']
    features = ['Age', 'Education-Num', 'Capital-Gain', 'Capital-Loss', 'Hours-per-week']
    target = 'Income'
    
    # Lê os dados
    df = pd.read_csv(input_file, names=names)

    # Exibe informações do dataframe original
    ShowInformationDataFrame(df, "Dataframe original")

    # Separa as features para normalização
    x = df.loc[:, features].values

    # Normalização Min-Max
    x_minmax = MinMaxScaler().fit_transform(x)
    normalized_minmax_df = pd.DataFrame(data=x_minmax, columns=features)
    normalized_minmax_df[target] = df[target]

    # Salva DataFrame normalizado em arquivo
    output_file_minmax = 'C:/MineracaoDados/adult/DataNormalization_MinMax.data'
    normalized_minmax_df.to_csv(output_file_minmax, index=False)
    
    # Gera gráficos para análise de medidas de posição relativa
    plot_comparative_violinplots(df, normalized_minmax_df, features)
    plot_comparative_histograms(df, normalized_minmax_df, features)
    
    # Gera gráfico de barras para medidas de associação
    plot_correlation_barchart(normalized_minmax_df, features, target)

def ShowInformationDataFrame(df, message=""):
    print(f"{message}\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")

def plot_comparative_violinplots(original_df, normalized_df, features):
    fig, axes = plt.subplots(nrows=2, ncols=len(features), figsize=(20, 10))
    
    for i, feature in enumerate(features):
        sns.violinplot(data=original_df, y=feature, ax=axes[0, i])
        axes[0, i].set_title(f'Original {feature}')
        
        sns.violinplot(data=normalized_df, y=feature, ax=axes[1, i])
        axes[1, i].set_title(f'Normalized {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_comparative_histograms(original_df, normalized_df, features):
    fig, axes = plt.subplots(nrows=2, ncols=len(features), figsize=(20, 10), sharey=True)
    
    for i, feature in enumerate(features):
        sns.histplot(original_df[feature], bins=30, kde=True, ax=axes[0, i])
        axes[0, i].set_title(f'Original {feature}')
        
        sns.histplot(normalized_df[feature], bins=30, kde=True, ax=axes[1, i])
        axes[1, i].set_title(f'Normalized {feature}')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_barchart(df, features, target):
    # Convert target to numerical values for correlation calculation
    df[target] = df[target].apply(lambda x: 1 if x == '>50K' else 0)

    # Calculate correlation
    correlations = df[features + [target]].corr()[target][:-1]

    # Plot correlation bar chart
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar', color='skyblue')
    plt.title('Correlation between Features and Income')
    plt.xlabel('Features')
    plt.ylabel('Correlation with Income')
    plt.show()

if __name__ == "__main__":
    main()
