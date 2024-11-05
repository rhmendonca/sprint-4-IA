import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Função para carregar dados do arquivo JSON
def carregar_dados(nome_arquivo):
    with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
        return json.load(arquivo)

# Carregar dados de celulares e usuários
usuarios = carregar_dados('usuarios.json')
celulares = carregar_dados('celulares.json')

# Função para encontrar um celular pelo modelo
def encontrar_celular_por_modelo(modelo, celulares):
    for celular in celulares:
        if celular['modelo'] == modelo:
            return celular
    return None

# Vetor de características dos celulares
def criar_vetor_celular(celular):
    sistema_operacional_vetor = {
        "Android": [1, 0],
        "iOS": [0, 1]
    }.get(celular['sistema_operacional'], [0, 0])
    
    # Vetor incluindo as características adicionais
    return np.array([
        celular['preco'], 
        celular['tamanho_tela'], 
        celular.get('armazenamento', 64),  # Default para 64GB caso não informado
        celular.get('ram', 4),  # Default para 4GB de RAM
        celular.get('camera', 12)  # Default para 12MP
    ] + sistema_operacional_vetor)

# Função para calcular o vetor médio das compras do usuário
def calcular_vetor_medio(usuario, celulares):
    historico_compras = usuario['historico_compras']
    
    vetor_medio = np.zeros(7)  # Ajustar com base no tamanho do vetor de características
    for modelo in historico_compras:
        celular = encontrar_celular_por_modelo(modelo, celulares)
        if celular:
            vetor_medio += criar_vetor_celular(celular)
    vetor_medio /= len(historico_compras) if historico_compras else 1  # Evita divisão por zero
    
    return vetor_medio

# Função para recomendar celulares com base no histórico de compras do usuário
def recomendar_celulares(usuario, celulares, top_n=5):
    # Definir faixa de preço com base na renda do usuário (30% abaixo e 50% acima)
    faixa_preco = (usuario['renda'] * 0.7, usuario['renda'] * 1.5)
    
    vetor_medio = calcular_vetor_medio(usuario, celulares)
    similaridades = []
    
    for celular in celulares:
        if celular['modelo'] not in usuario['historico_compras'] and faixa_preco[0] <= celular['preco'] <= faixa_preco[1]:
            vetor_celular = criar_vetor_celular(celular)
            similaridade = cosine_similarity([vetor_medio], [vetor_celular])[0][0]
            similaridades.append((celular, similaridade))
    
    # Ordenar celulares pela similaridade
    similaridades.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar os 'top_n' celulares mais semelhantes
    return [celular[0] for celular in similaridades[:top_n]]

# Função para recomendar celulares para um usuário específico
def recomendar_para_usuario(usuario_id, top_n=5):
    usuario = next((u for u in usuarios if u['id'] == usuario_id), None)
    
    if not usuario:
        print(f"Usuário com ID {usuario_id} não encontrado.")
        return
    
    print(f"Recomendações para o usuário {usuario['id']} (renda: {usuario['renda']}, idade: {usuario['idade']}):")
    recomendacoes = recomendar_celulares(usuario, celulares, top_n=top_n)
    for i, celular in enumerate(recomendacoes, start=1):
        print(f"{i}. {celular['marca']} {celular['modelo']} - Preço: R${celular['preco']}")

# Chamar a função de recomendação para um usuário específico
usuario_id = 2  # Troque esse valor pelo ID do usuário que deseja recomendar
recomendar_para_usuario(usuario_id, top_n=5)
