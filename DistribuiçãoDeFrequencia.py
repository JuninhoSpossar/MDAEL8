import pandas as pd

def calcular_distribuicao_de_frequencia(dados, coluna):
    frequencias = {}
    for dado in dados[coluna]:
        if dado in frequencias:
            frequencias[dado] += 1
        else:
            frequencias[dado] = 1
    return frequencias

# Ler o arquivo de dados
dados = pd.read_csv('C:\\MineraçãoDeDados\\adult\\adultmodificacao4')

# Lista de colunas para calcular a distribuição de frequência
colunas = ['Education-num', 'Marital-satus', 'Occupation', 'Relationship', 'Race', 'Capital-gain', 'Capital-loss', 'Hours-per-week', 'Native-Country', 'Income', 'Sex_Male', 'Age']

for coluna in colunas:
    # Calcular a distribuição de frequência
    distribuicao = calcular_distribuicao_de_frequencia(dados, coluna)

    # Imprimir a distribuição de frequência
    for dado, frequencia in distribuicao.items():
        print(f"Dado: {dado}, Frequência: {frequencia}")