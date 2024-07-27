import pandas as pd

# Ler o arquivo de dados
dados = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Lista de colunas para calcular a moda
colunas = ['Education-num', 'Marital-satus', 'Occupation', 'Relationship', 'Race', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-Country', 'Income', 'Sex_Male', 'Age']

for coluna in colunas:
    # Calcular a moda
    moda = dados[coluna].mode()[0]

    # Imprimir a moda
    print(f"Coluna: {coluna}, Moda: {moda}")