import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo de dados
data = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Calcular a matriz de correlação
correlation_matrix = data.corr()

# Imprimir a matriz de correlação
print(correlation_matrix)

# Plotar o gráfico de matriz de correlação
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()