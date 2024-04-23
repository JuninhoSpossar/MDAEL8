import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ler o arquivo de dados
data = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Calcular a correlação de todas as colunas com 'Income'
correlation_with_income = data.corrwith(data['Income'])

# Tirar 'Income' dos dados de correlação
correlation_with_income = correlation_with_income.drop('Income')

# Imprimir a correlação com 'Income'
print(correlation_with_income)

# Plotar o gráfico de correlação sem 'Income'
correlation_with_income.plot(kind='bar', color='blue')
plt.title('Correlação Income')
plt.show()
