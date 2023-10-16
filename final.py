import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Cria as pastas para armazenar as imagens
folders = ['heatmap', 'matrix', 'trends_and_seasonal_decomposition']
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


# Carregando os dados
data = pd.read_csv('weather_description.csv')

# Convertendo 'datetime' para datetime object e criando colunas para ano, mês, dia, hora e ano-mês
data['datetime'] = pd.to_datetime(data['datetime'])
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day
data['hour'] = data['datetime'].dt.hour
data['year_month'] = data['datetime'].dt.to_period('M')

def create_transition_matrix(city_column):
    unique_conditions = city_column.dropna().unique()
    transition_matrix = pd.DataFrame(0, index=unique_conditions, columns=unique_conditions)
    for i in range(len(city_column) - 1):
        current_condition = city_column.iloc[i]
        next_condition = city_column.iloc[i+1]
        if pd.notna(current_condition) and pd.notna(next_condition):
            transition_matrix.loc[current_condition, next_condition] += 1
    transition_matrix = transition_matrix.divide(transition_matrix.sum(axis=1), axis=0)
    return transition_matrix

def predict_next_state(current_state, transition_matrix, states):
    """
    Função para prever o próximo estado usando a matriz de transição.
    """
    probabilities = transition_matrix.loc[current_state].values
    return np.random.choice(states, p=probabilities)

def plot_matrix(matrix_data, city_name, matrix_type='transition'):
    plt.figure(figsize=(20, 15))
    if matrix_type == 'frequency':
        fmt_str = ".0f"
    else:
        fmt_str = ".2f"
    sns.heatmap(data=matrix_data, cmap='YlGnBu', annot=True, fmt=fmt_str)
    if matrix_type == 'transition':
        save_path = f'matrix/{city_name}_{matrix_type}_matrix.png'
    else:
        save_path = f'heatmap/{city_name}_{matrix_type}_matrix.png'
    plt.title(f'{matrix_type.capitalize()} Matrix for {city_name}')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_monthly_trends(city_data, city_name):
    city_data_grouped = city_data.groupby('year_month')[city_name].value_counts().unstack().fillna(0)
    monthly_totals = city_data_grouped.sum(axis=1)
    plt.figure(figsize=(12, 6))
    monthly_totals.plot()
    plt.title(f'Monthly Trend of Weather Conditions in {city_name}')
    plt.xlabel('Date')
    plt.ylabel('Total Conditions')
    plt.savefig(f'trends_and_seasonal_decomposition/{city_name}_monthly_trends.png', bbox_inches='tight')
    plt.close()

def plot_seasonal_decomposition(city_data, city_name):
    city_data_grouped = city_data.groupby('year_month')[city_name].value_counts().unstack().fillna(0)
    
    # Assegurando que todos os meses entre o mínimo e o máximo estejam presentes no dataset
    idx = pd.period_range(city_data_grouped.index.min(), city_data_grouped.index.max(), freq='M')
    city_data_grouped = city_data_grouped.reindex(idx, fill_value=0)
    
    monthly_totals = city_data_grouped.sum(axis=1)
    
    # Agora, removendo possíveis lacunas e preenchendo com zero
    monthly_totals = monthly_totals.asfreq('M', fill_value=0)
    
    # Adicionando a especificação de período
    result = seasonal_decompose(monthly_totals, model='additive', period=12)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 8))
    result.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    result.seasonal.plot(ax=ax3)    
    ax3.set_ylabel('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    plt.tight_layout()
    plt.savefig(f'trends_and_seasonal_decomposition/{city_name}_seasonal_decomposition.png', bbox_inches='tight')
    plt.close()

# Lista das cidades no seu dataset
cities = data.columns[1:-5]

results_list = []

for city in cities:
    city_data_by_hour = data.groupby('hour')[city].value_counts().unstack().fillna(0)
    plot_matrix(city_data_by_hour, city, matrix_type='frequency')
    transition_matrix = create_transition_matrix(data[city])
    plot_matrix(transition_matrix, city)
    plot_monthly_trends(data, city)
    plot_seasonal_decomposition(data, city)
    
    data_until_2016 = data[data['year'] < 2017]
    data_2017 = data[data['year'] == 2017]
    transition_matrix_until_2016 = create_transition_matrix(data_until_2016[city])
    
    states = transition_matrix_until_2016.index.tolist()

    # Abordagem 1: Prever 2017 com base apenas no último estado de 2016
    current_state = data_until_2016.iloc[-1][city]
    predictions_based_on_last = [predict_next_state(current_state, transition_matrix_until_2016, states) for _ in range(len(data_2017))]

    # Abordagem 2: Prever de forma sequencial, atualizando o estado atual a cada previsão
    current_state = data_until_2016.iloc[-1][city]
    predictions_sequential = []
    for _ in range(len(data_2017)):
        next_state = predict_next_state(current_state, transition_matrix_until_2016, states)
        predictions_sequential.append(next_state)
        current_state = next_state
    
    actual_data_2017 = data_2017[city].tolist()

    results_based_on_last = [1 if predictions_based_on_last[i] == actual_data_2017[i] else 0 for i in range(len(predictions_based_on_last))]
    results_sequential = [1 if predictions_sequential[i] == actual_data_2017[i] else 0 for i in range(len(predictions_sequential))]

    accuracy_based_on_2017 = sum(results_based_on_last) / len(results_based_on_last)
    accuracy_based_on_2013_2016 = sum(results_sequential) / len(results_sequential)
    
    new_row = {
        "City": city,
        "Accuracy_2017": accuracy_based_on_2017,
        "Accuracy_based_on_2013_2016": accuracy_based_on_2013_2016
    }
    results_list.append(new_row)
    
results_df = pd.DataFrame(results_list)

barWidth = 0.3

# Definindo as posições das barras
r1 = np.arange(len(results_df['City']))
r2 = [x + barWidth for x in r1]

plt.figure(figsize=(15,10))

# Criando as barras
plt.bar(r1, results_df['Accuracy_2017'], color='blue', width=barWidth, edgecolor='grey', label='Precisão baseada somente com dados de 2017')
plt.bar(r2, results_df['Accuracy_based_on_2013_2016'], color='orange', width=barWidth, edgecolor='grey', label='Precisão baseada com dados entre 2013-2016')

# Adicionando títulos e labels
plt.title('Comparação de Precisão entre as Duas Abordagens', fontweight='bold')
plt.xlabel('Cidades', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(results_df['City']))], results_df['City'], rotation=90)
plt.ylabel('Precisão', fontweight='bold')

# Criando a legenda e mostrando o gráfico
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_comparison.png')