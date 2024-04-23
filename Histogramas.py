import pandas as pd
import matplotlib.pyplot as plt

# Ler o arquivo CSV com o delimitador correto
data = pd.read_csv('C:\\MineraçãoDeDados\\b.csv', delimiter=';')

# Agrupar os dados por 'education' e 'income' e contar o número de ocorrências
grouped_data = data.groupby(['workclass', 'income']).size()

# Imprimir os resultados
print(grouped_data)

# Para visualizar os resultados em um gráfico de barras
grouped_data.unstack().plot(kind='bar', stacked=True)
plt.title('Distribuição de Income por WorckClass')
plt.xlabel('Age')
plt.ylabel('Frequência')
plt.show()