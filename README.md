# Datathon - Case Passos Mágicos

## Mudando a vida de crianças e jovens por meio da educação

Projeto de Data Analytics desenvolvido para o Datathon da Pós-Tech, analisando dados da **Associação Passos Mágicos** - organização com 32 anos de atuação transformando a vida de crianças e jovens de baixa renda em Embu-Guaçu/SP por meio da educação.

## Estrutura do Projeto

```
├── notebooks/
│   ├── 01_limpeza_eda.ipynb          # Limpeza de dados + Análise Exploratória (11 perguntas)
│   └── 02_modelo_preditivo.ipynb     # Modelo preditivo de risco de defasagem
├── app/
│   └── app.py                        # Aplicação Streamlit para deploy do modelo
├── models/
│   ├── best_model.pkl                # Modelo treinado (XGBoost)
│   ├── scaler.pkl                    # Scaler para normalização
│   ├── feature_names.pkl             # Nomes das features
│   └── model_metadata.json           # Metadados e métricas do modelo
├── outputs/                          # Visualizações geradas pela análise
├── run_analysis.py                   # Script executável da análise completa
├── generate_notebooks.py             # Gerador dos notebooks .ipynb
├── requirements.txt                  # Dependências Python
└── README.md
```

## Dataset

Base de dados PEDE (Pesquisa Extensiva do Desenvolvimento Educacional) contendo indicadores de 2022, 2023 e 2024:

| Indicador | Nome | Dimensão |
|-----------|------|----------|
| **IAN** | Adequação ao Nível | Acadêmica |
| **IDA** | Desempenho Acadêmico | Acadêmica |
| **IEG** | Engajamento | Acadêmica |
| **IAA** | Autoavaliação | Psicossocial |
| **IPS** | Psicossocial | Psicossocial |
| **IPP** | Psicopedagógico | Psicopedagógica |
| **IPV** | Ponto de Virada | Psicopedagógica |
| **INDE** | Desenvolvimento Educacional | Composto |

**Fórmula INDE (Fases 0-7):** `INDE = IAN×0.1 + IDA×0.2 + IEG×0.2 + IAA×0.1 + IPS×0.1 + IPP×0.1 + IPV×0.2`

## Perguntas Respondidas na Análise

1. **IAN** - Perfil de defasagem e evolução ao longo dos anos
2. **IDA** - Tendência do desempenho acadêmico por fases e anos
3. **IEG** - Relação entre engajamento, desempenho e ponto de virada
4. **IAA** - Coerência da autoavaliação com desempenho real
5. **IPS** - Padrões psicossociais que precedem quedas de desempenho
6. **IPP** - Avaliações psicopedagógicas vs defasagem (IAN)
7. **IPV** - Comportamentos que mais influenciam o ponto de virada
8. **Multidimensionalidade** - Combinações de indicadores que elevam o INDE
9. **Machine Learning** - Modelo preditivo de risco de defasagem
10. **Efetividade** - Impacto do programa nas classificações PEDRA
11. **Insights** - Análises por gênero, tempo na associação, instituição

## Modelo Preditivo

**Melhor modelo: XGBoost**

| Métrica | Valor |
|---------|-------|
| AUC-ROC | 0.987 |
| F1-Score | 0.953 |
| Acurácia | 94.1% |

O modelo identifica alunos em **risco de defasagem** com base em critérios combinados:
- IAN ≤ 5.0 (defasagem moderada ou severa)
- Defasagem < 0
- Classificação PEDRA = Quartzo

## Aplicação Streamlit

A aplicação permite:
- **Previsão Individual**: inserir indicadores de um aluno e obter probabilidade de risco
- **Previsão em Lote**: upload de CSV para análise em massa
- **Sobre o Modelo**: documentação completa da metodologia

### Executar localmente

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Como Reproduzir

```bash
# 1. Clonar o repositório
git clone https://github.com/rodrigonakatsu/tec5fiap.git
cd tec5fiap

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Extrair os dados
# Descompactar base_de_dados.zip na raiz do projeto

# 4. Rodar os notebooks (na ordem)
# Abrir no VSCode ou Jupyter:
# - notebooks/01_limpeza_eda.ipynb  → limpeza, EDA e geração do data_consolidado.csv
# - notebooks/02_modelo_preditivo.ipynb → treina e salva o modelo em models/

# Opcional: regenerar os notebooks a partir dos scripts
python generate_notebooks.py

# 5. Executar aplicação Streamlit
python -m streamlit run app/app.py
```

> **Atenção:** Os arquivos `models/*.pkl` já estão no repositório. Caso não queira rodar os notebooks, pode pular o passo 4 e ir direto para o passo 5.

## Tecnologias

- **Python 3.13**
- **Pandas / NumPy** - Manipulação de dados
- **Matplotlib / Seaborn** - Visualizações
- **Scikit-learn** - Machine Learning
- **XGBoost** - Gradient Boosting
- **Streamlit** - Aplicação web
- **Plotly** - Gráficos interativos

## Classificação PEDRA

| Pedra | Faixa INDE | Significado |
|-------|-----------|-------------|
| Quartzo | 2.405 - 5.506 | Abaixo da média |
| Ágata | 5.506 - 6.868 | Médio-baixo |
| Ametista | 6.868 - 8.230 | Médio-alto |
| Topázio | 8.230 - 9.294 | Acima da média |
