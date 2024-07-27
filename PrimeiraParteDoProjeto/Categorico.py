import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo CSV com o delimitador correto
data = pd.read_csv('C:\\MineraçãoDeDados\\b.csv', delimiter=';')

#features = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital_Status', 'Occupation',
           #  'Relationship', 'Race', 'Sex', 'Capital_Gain', 'Capital_Loss', 'Hours_per_Week', 'Native_Country','Income']

print(data.columns)
# Especificar as colunas que você quer plotar
colunas_para_plotar = ['income']

# Especificar o título do gráfico
titulo_do_grafico = 'Income'

# Criar gráficos de histograma para cada coluna especificada
for column in colunas_para_plotar:
    plt.figure()
    data[column].value_counts().plot(kind='bar')
    plt.title(f'{titulo_do_grafico} - {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.show()