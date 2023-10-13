import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Certifique-se de que uma pasta chamada 'heatmap' existe ou crie-a
if not os.path.exists('heatmap'):
    os.makedirs('heatmap')

# Carregando os dados
data = pd.read_csv('weather_description.csv')

# Convertendo 'datetime' para datetime object e criando colunas para ano, mês, dia e hora
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour

def create_transition_matrix(city_column):
    """
    Função para criar uma matriz de transição para uma coluna de cidade.
    
    Args:
    - city_column: Series com dados da cidade.

    Retorna:
    - DataFrame representando a matriz de transição.
    """
    # Criando um dataframe vazio para a matriz de transição
    unique_conditions = city_column.dropna().unique()
    transition_matrix = pd.DataFrame(0, index=unique_conditions, columns=unique_conditions)
    
    # Preenchendo a matriz com contagens de transições
    for i in range(len(city_column) - 1):
        current_condition = city_column.iloc[i]
        next_condition = city_column.iloc[i+1]
        if pd.notna(current_condition) and pd.notna(next_condition):
            transition_matrix.loc[current_condition, next_condition] += 1
            
    # Normalizando a matriz para obter probabilidades
    transition_matrix = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
    
    return transition_matrix

def plot_heatmap(matrix_data, city_name, matrix_type='transition'):
    """
    Função para plotar heatmap.
    
    Args:
    - matrix_data: DataFrame com os dados a serem plotados.
    - city_name: Nome da cidade.
    - matrix_type: Tipo da matriz para o título (padrão é 'transition').
    """
    if matrix_type == 'frequency':
        plt.figure(figsize=(20, 15))  # Aumentando o tamanho para a matriz de frequência 20 = largura 15 = altura (em polegadas)
        fmt_str = ".0f"
    else:
        plt.figure(figsize=(20, 15))
        fmt_str = ".2f"
        
    sns.heatmap(data=matrix_data, cmap='YlGnBu', annot=True, fmt=fmt_str)
    plt.title(f'{matrix_type.capitalize()} Matrix for {city_name}')
    
    # Salvando a figura em um arquivo dentro da pasta 'heatmap'
    plt.savefig(f'heatmap/{city_name}_{matrix_type}_matrix.png', bbox_inches='tight')
    plt.close()  # Fecha a figura para liberar memória



# Lista das cidades no seu dataset
cities = data.columns[1:-4]  # Estou supondo que as últimas 4 colunas são datetime, year, month e hour.

for city in cities:
    city_data_by_hour = data.groupby('hour')[city].value_counts().unstack().fillna(0)
    plot_heatmap(city_data_by_hour, city, matrix_type='frequency')
    
    transition_matrix = create_transition_matrix(data[city])
    plot_heatmap(transition_matrix, city)
