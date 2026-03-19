"""
Datathon - Passos Mágicos
Aplicação Streamlit para Modelo Preditivo de Risco de Defasagem

Deploy: Streamlit Community Cloud
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.graph_objects as go

# --- Configuração da Página ---
st.set_page_config(
    page_title="Passos Mágicos - Previsão de Risco",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')


@st.cache_resource
def load_model():
    """Carrega o modelo treinado e artefatos."""
    model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    features_path = os.path.join(MODELS_DIR, 'feature_names.pkl')
    metadata_path = os.path.join(MODELS_DIR, 'model_metadata.json')

    if not os.path.exists(model_path):
        return None, None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return model, scaler, features, metadata


def create_gauge_chart(probability):
    """Cria gráfico de gauge para probabilidade de risco."""
    if probability < 0.3:
        color = "#2ecc71"
        nivel = "BAIXO"
    elif probability < 0.6:
        color = "#f39c12"
        nivel = "MODERADO"
    else:
        color = "#e74c3c"
        nivel = "ALTO"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risco de Defasagem: {nivel}", 'font': {'size': 24}},
        number={'suffix': '%', 'font': {'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d5f5e3'},
                {'range': [30, 60], 'color': '#fdebd0'},
                {'range': [60, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=350, margin=dict(t=80, b=20, l=20, r=20))
    return fig


def create_radar_chart(values, labels):
    """Cria gráfico radar dos indicadores do aluno."""
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Aluno',
        fillcolor='rgba(52, 152, 219, 0.3)',
        line=dict(color='#3498db', width=2)
    ))

    # Referência: média ideal (7.0)
    ref_values = [7.0] * len(labels)
    fig.add_trace(go.Scatterpolar(
        r=ref_values + [ref_values[0]],
        theta=labels + [labels[0]],
        fill='toself',
        name='Referência (7.0)',
        fillcolor='rgba(46, 204, 113, 0.1)',
        line=dict(color='#2ecc71', width=1, dash='dash')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        height=400,
        margin=dict(t=40, b=40, l=60, r=60)
    )
    return fig


# --- SIDEBAR ---
st.sidebar.title("Passos Mágicos")
st.sidebar.markdown("### Modelo Preditivo de Risco de Defasagem")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    ["Previsão Individual", "Sobre o Modelo"]
)

# --- Carregar modelo ---
model, scaler, feature_names, metadata = load_model()

if model is None:
    st.error("""
    **Modelo não encontrado!**

    Execute primeiro o notebook `02_modelo_preditivo.ipynb` para treinar e salvar o modelo.

    Os arquivos esperados em `models/`:
    - `best_model.pkl`
    - `scaler.pkl`
    - `feature_names.pkl`
    - `model_metadata.json`
    """)
    st.stop()

# Identificar features base (indicadores PEDE)
INDICADORES_BASE = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'INDE', 'IAN']
INDICADORES_DESCRICAO = {
    'IDA': 'Indicador de Desempenho Acadêmico (0-10)',
    'IEG': 'Indicador de Engajamento (0-10)',
    'IAA': 'Indicador de Autoavaliação (0-10)',
    'IPS': 'Indicador Psicossocial (0-10)',
    'IPP': 'Indicador Psicopedagógico (0-10)',
    'IPV': 'Indicador de Ponto de Virada (0-10)',
    'INDE': 'Índice de Desenvolvimento Educacional (0-10)',
    'IAN': 'Indicador de Adequação ao Nível (2.5, 5.0 ou 10.0)',
    'Fase_num': 'Fase do aluno no programa (0-8)',
    'Idade_num': 'Idade do aluno',
    'Anos_PM': 'Anos na Passos Mágicos',
    'IDA_x_IEG': 'Interação: IDA × IEG',
    'IPS_x_IAA': 'Interação: IPS × IAA',
    'Media_Academica': 'Média dos indicadores acadêmicos',
    'Media_Psico': 'Média dos indicadores psicossociais',
}

# =============================
# PÁGINA 1: PREVISÃO INDIVIDUAL
# =============================
if page == "Previsão Individual":
    st.title("🔮 Previsão de Risco de Defasagem")
    st.markdown("""
    Insira os indicadores do aluno para calcular a **probabilidade de estar em risco de defasagem**.
    """)

    st.markdown("---")

    # Formulário de entrada
    col1, col2, col3 = st.columns(3)

    input_values = {}

    with col1:
        st.subheader("📚 Indicadores Acadêmicos")
        if 'IDA' in feature_names:
            input_values['IDA'] = st.number_input("IDA - Desempenho Acadêmico", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        if 'IEG' in feature_names:
            input_values['IEG'] = st.number_input("IEG - Engajamento", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if 'IAN' in feature_names:
            input_values['IAN'] = st.select_slider("IAN - Adequação ao Nível",
                                                     options=[2.5, 5.0, 10.0], value=5.0)
        if 'INDE' in feature_names:
            input_values['INDE'] = st.number_input("INDE - Desenvolvimento Educacional", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    with col2:
        st.subheader("🧠 Indicadores Psicossociais")
        if 'IAA' in feature_names:
            input_values['IAA'] = st.number_input("IAA - Autoavaliação", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if 'IPS' in feature_names:
            input_values['IPS'] = st.number_input("IPS - Psicossocial", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if 'IPP' in feature_names:
            input_values['IPP'] = st.number_input("IPP - Psicopedagógico", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        if 'IPV' in feature_names:
            input_values['IPV'] = st.number_input("IPV - Ponto de Virada", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

    with col3:
        st.subheader("📋 Dados Adicionais")
        if 'Fase_num' in feature_names:
            input_values['Fase_num'] = st.selectbox("Fase", options=list(range(0, 9)), index=3)
        if 'Idade_num' in feature_names:
            input_values['Idade_num'] = st.number_input("Idade", min_value=6, max_value=25, value=12)
        if 'Anos_PM' in feature_names:
            input_values['Anos_PM'] = st.number_input("Anos na Passos Mágicos", min_value=0, max_value=10, value=2)

    # Calcular features derivadas
    if 'IDA_x_IEG' in feature_names and 'IDA' in input_values and 'IEG' in input_values:
        input_values['IDA_x_IEG'] = input_values['IDA'] * input_values['IEG']
    if 'IPS_x_IAA' in feature_names and 'IPS' in input_values and 'IAA' in input_values:
        input_values['IPS_x_IAA'] = input_values['IPS'] * input_values['IAA']
    if 'Media_Academica' in feature_names:
        acad = [input_values.get(k, 0) for k in ['IDA', 'IEG', 'IAA'] if k in input_values]
        input_values['Media_Academica'] = np.mean(acad) if acad else 0
    if 'Media_Psico' in feature_names:
        psico = [input_values.get(k, 0) for k in ['IPS', 'IPP'] if k in input_values]
        input_values['Media_Psico'] = np.mean(psico) if psico else 0

    st.markdown("---")

    # Botão de previsão
    if st.button("🔍 Calcular Risco", type="primary", use_container_width=True):
        # Montar input
        input_df = pd.DataFrame([{f: input_values.get(f, 0) for f in feature_names}])

        # Verificar se modelo precisa de dados escalados
        model_type = metadata.get('model_name', '')
        if 'Logistic' in model_type:
            input_array = scaler.transform(input_df)
        else:
            input_array = input_df.values

        # Predição
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]

        # Exibir resultados
        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

        with col_result2:
            # Radar chart com indicadores base
            radar_labels = [k for k in INDICADORES_BASE if k in input_values]
            radar_values = [input_values[k] for k in radar_labels]
            if radar_labels:
                st.plotly_chart(create_radar_chart(radar_values, radar_labels), use_container_width=True)

        # Interpretação
        st.markdown("---")
        if probability >= 0.6:
            st.error(f"""
            ⚠️ **ALERTA: Alto risco de defasagem ({probability*100:.1f}%)**

            **Recomendações:**
            - Acompanhamento psicopedagógico intensivo
            - Reforço acadêmico em disciplinas com maior defasagem
            - Apoio psicossocial para o aluno e família
            - Monitoramento semanal dos indicadores
            """)
        elif probability >= 0.3:
            st.warning(f"""
            ⚡ **ATENÇÃO: Risco moderado de defasagem ({probability*100:.1f}%)**

            **Recomendações:**
            - Acompanhamento regular dos indicadores
            - Atividades de engajamento adicionais
            - Avaliação psicossocial periódica
            """)
        else:
            st.success(f"""
            ✅ **Risco baixo de defasagem ({probability*100:.1f}%)**

            O aluno apresenta indicadores saudáveis. Manter acompanhamento regular.
            """)

# =============================
# PÁGINA 3: SOBRE O MODELO
# =============================
elif page == "Sobre o Modelo":
    st.title("ℹ️ Sobre o Modelo Preditivo")

    st.markdown("""
    ## Objetivo
    Identificar alunos em **risco de defasagem educacional** com base nos indicadores do PEDE
    (Pesquisa Extensiva do Desenvolvimento Educacional) da Associação Passos Mágicos.

    ## Metodologia
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Definição de Risco")
        st.markdown("""
        Um aluno é classificado como **em risco** quando atende pelo menos um critério:
        - **IAN ≤ 5.0** → Defasagem moderada ou severa
        - **Defasagem < 0** → Abaixo do nível ideal
        - **Pedra = Quartzo** → Classificação mais baixa do INDE
        """)

    with col2:
        st.markdown("### Métricas do Modelo")
        if metadata:
            metrics = metadata.get('metrics', {})
            st.metric("Modelo", metadata.get('model_name', 'N/A'))
            st.metric("AUC-ROC", f"{metrics.get('auc', 0):.3f}")
            st.metric("F1-Score", f"{metrics.get('f1', 0):.3f}")
            st.metric("Acurácia", f"{metrics.get('accuracy', 0):.3f}")

    st.markdown("---")

    st.markdown("### Features Utilizadas")
    features_info = []
    for f in feature_names:
        desc = INDICADORES_DESCRICAO.get(f, f)
        features_info.append({"Feature": f, "Descrição": desc})
    st.table(pd.DataFrame(features_info))

    st.markdown("""
    ---
    ### Indicadores PEDE

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

    ### Fórmula do INDE (Fases 0-7)
    ```
    INDE = IAN×0.1 + IDA×0.2 + IEG×0.2 + IAA×0.1 + IPS×0.1 + IPP×0.1 + IPV×0.2
    ```

    ### Classificação PEDRA
    | Pedra | Faixa INDE |
    |-------|-----------|
    | Quartzo | 2.405 a 5.506 |
    | Ágata | 5.506 a 6.868 |
    | Ametista | 6.868 a 8.230 |
    | Topázio | 8.230 a 9.294 |
    """)

    st.markdown("""
    ---
    ### Sobre a Passos Mágicos
    A Associação Passos Mágicos tem 32 anos de atuação, trabalhando na transformação da vida
    de crianças e jovens de baixa renda do município de Embu-Guaçu, SP, por meio da educação.

    *Projeto desenvolvido para o Datathon - Pós-Tech Data Analytics*
    """)

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    "*Datathon - Passos Mágicos*<br>"
    "*Pós-Tech Data Analytics*",
    unsafe_allow_html=True
)
