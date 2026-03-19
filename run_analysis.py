"""
Script para executar a análise completa e gerar os artefatos do modelo.
Este script replica a lógica dos notebooks sem depender do Jupyter.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import joblib
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, accuracy_score)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost não disponível, usando GradientBoosting")

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.figsize'] = (12, 6)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(BASE_DIR, 'BASE DE DADOS PEDE 2024 - DATATHON.xlsx')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

CORES_PEDRA = {
    'Quartzo': '#E8D5B7',
    'Ágata': '#B8A9C9',
    'Ametista': '#9B59B6',
    'Topázio': '#F39C12'
}

# ========================================
# 1. CARREGAMENTO E LIMPEZA
# ========================================
print("=" * 60)
print("1. CARREGAMENTO E LIMPEZA DOS DADOS")
print("=" * 60)

df_2022 = pd.read_excel(XLSX_PATH, sheet_name='PEDE2022')
df_2023 = pd.read_excel(XLSX_PATH, sheet_name='PEDE2023')
df_2024 = pd.read_excel(XLSX_PATH, sheet_name='PEDE2024')

print(f"PEDE 2022: {df_2022.shape}")
print(f"PEDE 2023: {df_2023.shape}")
print(f"PEDE 2024: {df_2024.shape}")

# Exibir colunas
for nome, df_temp in [('2022', df_2022), ('2023', df_2023), ('2024', df_2024)]:
    print(f"\nColunas PEDE {nome}: {list(df_temp.columns)}")


def encontrar_coluna(df, patterns):
    """Encontra coluna no DataFrame por padrão."""
    for col in df.columns:
        col_lower = col.lower().strip()
        for pat in patterns:
            if pat.lower() in col_lower:
                return col
    return None


def get_indicador(df, indicador, ano=None):
    """Busca a coluna de um indicador no DataFrame."""
    suffix = str(ano)[-2:] if ano else ''
    patterns = []
    if ano:
        patterns = [f'{indicador} {suffix}', f'{indicador}_{ano}', f'{indicador}{suffix}']
    patterns.append(indicador)
    col = encontrar_coluna(df, patterns)
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors='coerce')
    return None


def padronizar_fase(val):
    """Converte valor de fase para numérico."""
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip().upper()
    fase_map = {'ALFA': 0, 'ALFABETIZAÇÃO': 0, 'ALFABETIZACAO': 0}
    if val_str in fase_map:
        return fase_map[val_str]
    for i in range(9):
        if f'FASE {i}' in val_str or val_str == str(i):
            return i
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# Construir dataset consolidado
records = []
for ano, df_src in [(2022, df_2022), (2023, df_2023), (2024, df_2024)]:
    suffix = str(ano)[-2:]

    inde = get_indicador(df_src, 'INDE', ano)
    ian = get_indicador(df_src, 'IAN', ano)
    ida = get_indicador(df_src, 'IDA', ano)
    ieg = get_indicador(df_src, 'IEG', ano)
    iaa = get_indicador(df_src, 'IAA', ano)
    ips = get_indicador(df_src, 'IPS', ano)
    ipp = get_indicador(df_src, 'IPP', ano)
    ipv = get_indicador(df_src, 'IPV', ano)

    # Buscar colunas auxiliares
    pedra_col = None
    for p in [f'Pedra {suffix}', f'Pedra {ano}', f'Pedra{suffix}']:
        if p in df_src.columns:
            pedra_col = p
            break

    fase_col = encontrar_coluna(df_src, ['fase'])
    defas_col = encontrar_coluna(df_src, ['defasagem', 'defas'])
    genero_col = encontrar_coluna(df_src, ['genero', 'gênero', 'sexo'])
    idade_col = encontrar_coluna(df_src, ['idade'])
    ingresso_col = encontrar_coluna(df_src, ['ingresso'])
    inst_col = encontrar_coluna(df_src, ['instituic'])
    pv_col = encontrar_coluna(df_src, ['atingiu pv', 'ponto_virada', 'atingiu'])
    ra_col = encontrar_coluna(df_src, ['ra', 'nome'])

    for idx in range(len(df_src)):
        rec = {
            'Ano': ano,
            'INDE': inde.iloc[idx] if inde is not None else np.nan,
            'IAN': ian.iloc[idx] if ian is not None else np.nan,
            'IDA': ida.iloc[idx] if ida is not None else np.nan,
            'IEG': ieg.iloc[idx] if ieg is not None else np.nan,
            'IAA': iaa.iloc[idx] if iaa is not None else np.nan,
            'IPS': ips.iloc[idx] if ips is not None else np.nan,
            'IPP': ipp.iloc[idx] if ipp is not None else np.nan,
            'IPV': ipv.iloc[idx] if ipv is not None else np.nan,
        }

        if pedra_col:
            rec['Pedra'] = df_src[pedra_col].iloc[idx]
        if defas_col:
            rec['Defasagem'] = pd.to_numeric(df_src[defas_col].iloc[idx], errors='coerce')
        if fase_col:
            rec['Fase'] = padronizar_fase(df_src[fase_col].iloc[idx])
        if genero_col:
            g = str(df_src[genero_col].iloc[idx]).strip()
            rec['Genero'] = {'Menina': 'Feminino', 'Menino': 'Masculino', 'menina': 'Feminino', 'menino': 'Masculino'}.get(g, g)
        if idade_col:
            rec['Idade'] = pd.to_numeric(df_src[idade_col].iloc[idx], errors='coerce')
        if ingresso_col:
            rec['Ano_Ingresso'] = pd.to_numeric(df_src[ingresso_col].iloc[idx], errors='coerce')
        if inst_col:
            rec['Instituicao'] = df_src[inst_col].iloc[idx]
        if pv_col:
            rec['Ponto_Virada'] = df_src[pv_col].iloc[idx]
        if ra_col:
            rec['Aluno_ID'] = df_src[ra_col].iloc[idx]

        records.append(rec)

df = pd.DataFrame(records)

# Padronizar Pedra
df['Pedra'] = df['Pedra'].replace({
    'Agata': 'Ágata', 'AGATA': 'Ágata', 'Ágata': 'Ágata',
    'Ametista': 'Ametista', 'AMETISTA': 'Ametista',
    'Quartzo': 'Quartzo', 'QUARTZO': 'Quartzo',
    'Topazio': 'Topázio', 'TOPAZIO': 'Topázio', 'Topázio': 'Topázio',
    'INCLUIR': np.nan
})

print(f"\nDataset consolidado: {df.shape}")
print(f"Distribuição por ano:\n{df['Ano'].value_counts().sort_index()}")
print(f"\nIndicadores - valores não nulos (ANTES da imputação):")
for col in ['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']:
    print(f"  {col}: {df[col].notna().sum()}")

# ========================================
# 1.5 KNN IMPUTER - PREENCHER IPP DE 2022
# ========================================
print("\n" + "-" * 50)
print("KNN IMPUTER: Preenchendo IPP faltante de 2022")
print("-" * 50)
print(f"IPP nulos por ano ANTES:")
print(df.groupby('Ano')['IPP'].apply(lambda x: x.isna().sum()).to_dict())

# Usar os outros indicadores como base para imputação do IPP
# O KNN encontra os vizinhos mais próximos (alunos com perfil similar)
# nos dados de 2023 e 2024 que TÊM IPP, e usa para preencher 2022
indicadores_impute = ['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPV', 'IPP']
df_impute = df[indicadores_impute].copy()

# Aplicar KNN Imputer (k=5 vizinhos)
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
df_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df_impute),
    columns=indicadores_impute,
    index=df_impute.index
)

# Substituir apenas o IPP que era nulo
mask_ipp_null = df['IPP'].isna()
df.loc[mask_ipp_null, 'IPP'] = df_imputed.loc[mask_ipp_null, 'IPP']

print(f"\nIPP nulos por ano DEPOIS:")
print(df.groupby('Ano')['IPP'].apply(lambda x: x.isna().sum()).to_dict())
print(f"\nIPP médio imputado (2022): {df[df['Ano']==2022]['IPP'].mean():.2f}")
print(f"IPP médio real (2023):     {df[df['Ano']==2023]['IPP'].mean():.2f}")
print(f"IPP médio real (2024):     {df[df['Ano']==2024]['IPP'].mean():.2f}")

# Recalcular INDE para 2022 com o IPP imputado
# INDE (Fases 0-7) = IAN×0.1 + IDA×0.2 + IEG×0.2 + IAA×0.1 + IPS×0.1 + IPP×0.1 + IPV×0.2
mask_2022 = df['Ano'] == 2022
df.loc[mask_2022, 'INDE_recalculado'] = (
    df.loc[mask_2022, 'IAN'] * 0.1 +
    df.loc[mask_2022, 'IDA'] * 0.2 +
    df.loc[mask_2022, 'IEG'] * 0.2 +
    df.loc[mask_2022, 'IAA'] * 0.1 +
    df.loc[mask_2022, 'IPS'] * 0.1 +
    df.loc[mask_2022, 'IPP'] * 0.1 +
    df.loc[mask_2022, 'IPV'] * 0.2
)
print(f"\nINDE original 2022: {df.loc[mask_2022, 'INDE'].mean():.2f}")
print(f"INDE recalculado com IPP: {df.loc[mask_2022, 'INDE_recalculado'].mean():.2f}")

print(f"\nIndicadores - valores não nulos (APÓS imputação):")
for col in ['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']:
    print(f"  {col}: {df[col].notna().sum()}")

# Salvar CSV consolidado
df.to_csv(os.path.join(BASE_DIR, 'data_consolidado.csv'), index=False)
print("\nDados consolidados salvos em data_consolidado.csv")

# ========================================
# 2. ANÁLISES EDA (Perguntas 1-11)
# ========================================
print("\n" + "=" * 60)
print("2. ANÁLISE EXPLORATÓRIA - GERANDO VISUALIZAÇÕES")
print("=" * 60)

pedra_ordem = ['Quartzo', 'Ágata', 'Ametista', 'Topázio']

# --- Pergunta 1: IAN ---
print("\nPergunta 1: IAN - Adequação ao Nível")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ano in enumerate([2022, 2023, 2024]):
    df_ano = df[df['Ano'] == ano]
    ian_vals = df_ano['IAN'].dropna()
    classif = ian_vals.apply(lambda v: 'Adequado' if v >= 9 else ('Moderada' if v >= 4 else 'Severa'))
    contagem = classif.value_counts()
    cores = {'Adequado': '#2ecc71', 'Moderada': '#f39c12', 'Severa': '#e74c3c'}
    bars = axes[i].bar(contagem.index, contagem.values, color=[cores.get(x, '#95a5a6') for x in contagem.index])
    axes[i].set_title(f'Classificação IAN - {ano}', fontweight='bold')
    axes[i].set_ylabel('Número de Alunos')
    total = contagem.sum()
    for bar, val in zip(bars, contagem.values):
        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{val}\n({val/total*100:.1f}%)', ha='center', fontweight='bold')
plt.suptitle('Perfil de Defasagem dos Alunos (IAN)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '01_ian_perfil_defasagem.png'), dpi=150, bbox_inches='tight')
plt.close()

# IAN evolução
fig, ax = plt.subplots(figsize=(10, 6))
ian_por_ano = df.groupby('Ano')['IAN'].agg(['mean', 'median', 'std']).round(2)
ax.plot(ian_por_ano.index, ian_por_ano['mean'], 'o-', color='#3498db', linewidth=2, markersize=10, label='Média')
ax.plot(ian_por_ano.index, ian_por_ano['median'], 's--', color='#e74c3c', linewidth=2, markersize=8, label='Mediana')
ax.fill_between(ian_por_ano.index, ian_por_ano['mean'] - ian_por_ano['std'],
                ian_por_ano['mean'] + ian_por_ano['std'], alpha=0.2, color='#3498db')
ax.set_xlabel('Ano'); ax.set_ylabel('IAN')
ax.set_title('Evolução do IAN Médio ao Longo dos Anos', fontweight='bold')
ax.legend(); ax.set_xticks(ian_por_ano.index); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '02_ian_evolucao.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos IAN salvos.")

# --- Pergunta 2: IDA ---
print("Pergunta 2: IDA - Desempenho Acadêmico")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ida_por_ano = df.groupby('Ano')['IDA'].agg(['mean', 'median', 'std', 'count']).round(2)
axes[0].bar(ida_por_ano.index, ida_por_ano['mean'], color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='white')
for ano_val, row in ida_por_ano.iterrows():
    axes[0].text(ano_val, row['mean'] + 0.05, f"{row['mean']:.2f}", ha='center', fontweight='bold', fontsize=13)
axes[0].set_xlabel('Ano'); axes[0].set_ylabel('IDA Médio')
axes[0].set_title('Desempenho Acadêmico Médio por Ano', fontweight='bold')
axes[0].set_xticks(ida_por_ano.index)

df_ida_fase = df.dropna(subset=['IDA', 'Fase'])
df_ida_fase = df_ida_fase[df_ida_fase['Fase'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
for ano in [2022, 2023, 2024]:
    dados = df_ida_fase[df_ida_fase['Ano'] == ano].groupby('Fase')['IDA'].mean()
    axes[1].plot(dados.index, dados.values, 'o-', label=str(ano), linewidth=2, markersize=8)
axes[1].set_xlabel('Fase'); axes[1].set_ylabel('IDA Médio')
axes[1].set_title('IDA Médio por Fase e Ano', fontweight='bold')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '04_ida_evolucao.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos IDA salvos.")

# --- Pergunta 3: IEG ---
print("Pergunta 3: IEG - Engajamento")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# IEG vs IDA
df_corr = df.dropna(subset=['IEG', 'IDA'])
axes[0].scatter(df_corr['IEG'], df_corr['IDA'], alpha=0.3, c='#3498db', s=15)
corr_val = df_corr['IEG'].corr(df_corr['IDA'])
z = np.polyfit(df_corr['IEG'].values, df_corr['IDA'].values, 1)
p = np.poly1d(z)
x_line = np.linspace(df_corr['IEG'].min(), df_corr['IEG'].max(), 100)
axes[0].plot(x_line, p(x_line), 'r--', linewidth=2)
axes[0].set_title(f'IEG vs IDA (r = {corr_val:.3f})', fontweight='bold')
axes[0].set_xlabel('IEG (Engajamento)'); axes[0].set_ylabel('IDA (Desempenho)')

# Matriz de correlação
corr_matrix = df[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'INDE']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, vmin=-1, vmax=1, ax=axes[1], square=True, linewidths=1)
axes[1].set_title('Correlação entre Indicadores', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '06_ieg_correlacoes.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos IEG salvos.")

# --- Pergunta 4: IAA ---
print("Pergunta 4: IAA - Autoavaliação")
df_disc = df.dropna(subset=['IAA', 'IDA']).copy()
df_disc['Discrepancia'] = df_disc['IAA'] - df_disc['IDA']
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df_disc['Discrepancia'], bins=40, color='#3498db', edgecolor='white', alpha=0.8)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Sem discrepância')
ax.set_title('Distribuição da Discrepância (IAA - IDA)', fontweight='bold')
ax.set_xlabel('IAA - IDA'); ax.set_ylabel('Frequência'); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '08_iaa_coerencia.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos IAA salvos.")

# --- Pergunta 5: IPS ---
print("Pergunta 5: IPS - Psicossocial")
df_ips = df.dropna(subset=['IPS', 'IDA', 'IEG']).copy()
df_ips['Faixa_IPS'] = pd.cut(df_ips['IPS'], bins=[0, 3, 5, 7, 10],
                              labels=['Baixo (0-3)', 'Médio-Baixo (3-5)', 'Médio-Alto (5-7)', 'Alto (7-10)'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
faixa_ida = df_ips.groupby('Faixa_IPS', observed=True)['IDA'].agg(['mean', 'std', 'count'])
axes[0].bar(range(len(faixa_ida)), faixa_ida['mean'], yerr=faixa_ida['std'], capsize=5,
            color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='white')
axes[0].set_xticks(range(len(faixa_ida))); axes[0].set_xticklabels(faixa_ida.index, rotation=45, ha='right')
axes[0].set_title('IDA Médio por Faixa de IPS', fontweight='bold'); axes[0].set_ylabel('IDA Médio')
faixa_ieg = df_ips.groupby('Faixa_IPS', observed=True)['IEG'].agg(['mean', 'std', 'count'])
axes[1].bar(range(len(faixa_ieg)), faixa_ieg['mean'], yerr=faixa_ieg['std'], capsize=5,
            color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], edgecolor='white')
axes[1].set_xticks(range(len(faixa_ieg))); axes[1].set_xticklabels(faixa_ieg.index, rotation=45, ha='right')
axes[1].set_title('IEG Médio por Faixa de IPS', fontweight='bold'); axes[1].set_ylabel('IEG Médio')
plt.suptitle('Impacto dos Aspectos Psicossociais', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '10_ips_impacto.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos IPS salvos.")

# --- Pergunta 6: IPP ---
print("Pergunta 6: IPP - Psicopedagógico")
df_pp = df.dropna(subset=['IPP', 'IAN']).copy()
if len(df_pp) > 10:
    fig, ax = plt.subplots(figsize=(10, 6))
    def classificar_ian(val):
        if val >= 9: return 'Adequado'
        elif val >= 4: return 'Moderada'
        else: return 'Severa'
    df_pp['IAN_Nivel'] = df_pp['IAN'].apply(classificar_ian)
    ordem = ['Severa', 'Moderada', 'Adequado']
    sns.boxplot(data=df_pp, x='IAN_Nivel', y='IPP', order=ordem,
                palette=['#e74c3c', '#f39c12', '#2ecc71'], ax=ax)
    ax.set_title('IPP por Nível de Defasagem (IAN)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, '12_ipp_vs_ian.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Gráficos IPP salvos.")
else:
    print("  Poucos dados de IPP para análise.")

# --- Pergunta 10: Efetividade ---
print("Pergunta 10: Efetividade do Programa")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ano in enumerate([2022, 2023, 2024]):
    df_ano = df[(df['Ano'] == ano) & (df['Pedra'].isin(pedra_ordem))]
    if len(df_ano) > 0:
        contagem = df_ano['Pedra'].value_counts().reindex(pedra_ordem).fillna(0)
        cores = [CORES_PEDRA.get(p, '#95a5a6') for p in contagem.index]
        bars = axes[i].bar(contagem.index, contagem.values, color=cores, edgecolor='gray')
        total = contagem.sum()
        for bar, val in zip(bars, contagem.values):
            if val > 0:
                axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                            f'{int(val)}\n({val/total*100:.1f}%)', ha='center', fontweight='bold')
        axes[i].set_title(f'PEDRA - {ano}', fontweight='bold'); axes[i].set_ylabel('Alunos')
plt.suptitle('Classificação dos Alunos por Pedra ao Longo dos Anos', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '17_pedra_evolucao.png'), dpi=150, bbox_inches='tight')
plt.close()

# Evolução de todos os indicadores
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes_flat = axes.flatten()
for i, ind in enumerate(['INDE', 'IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']):
    dados = df.groupby('Ano')[ind].agg(['mean', 'std']).dropna()
    if len(dados) > 0:
        axes_flat[i].bar(dados.index, dados['mean'], yerr=dados['std'], capsize=5,
                         color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='white')
        for j, (a, row) in enumerate(dados.iterrows()):
            axes_flat[i].text(a, row['mean'] + 0.05, f"{row['mean']:.2f}", ha='center', fontweight='bold')
        axes_flat[i].set_title(f'{ind}', fontweight='bold'); axes_flat[i].set_xticks(dados.index)
plt.suptitle('Evolução de Todos os Indicadores (2022-2024)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '18_evolucao_indicadores.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Gráficos de efetividade salvos.")

# --- Pergunta 11: Insights ---
print("Pergunta 11: Insights adicionais")
# Heatmap por Pedra
df_pedra_ind = df[df['Pedra'].isin(pedra_ordem)]
if len(df_pedra_ind) > 0:
    fig, ax = plt.subplots(figsize=(12, 6))
    heatmap_data = df_pedra_ind.groupby('Pedra')[['IAN', 'IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV']].mean()
    heatmap_data = heatmap_data.reindex(pedra_ordem)
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=1, ax=ax, vmin=0, vmax=10)
    ax.set_title('Perfil de Indicadores por Classificação PEDRA', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, '23_heatmap_pedra.png'), dpi=150, bbox_inches='tight')
    plt.close()

# Análise por gênero
df_genero = df[df['Genero'].isin(['Feminino', 'Masculino'])]
if len(df_genero) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    gen_inde = df_genero.groupby(['Ano', 'Genero'])['INDE'].mean().unstack()
    gen_inde.plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'], edgecolor='white')
    ax.set_title('INDE Médio por Gênero e Ano', fontweight='bold')
    ax.set_ylabel('INDE Médio'); ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, '20_analise_genero.png'), dpi=150, bbox_inches='tight')
    plt.close()
print("  Gráficos de insights salvos.")

# ========================================
# 3. MODELO PREDITIVO
# ========================================
print("\n" + "=" * 60)
print("3. MODELO PREDITIVO")
print("=" * 60)

# Criar target
def criar_target(row):
    em_risco = False
    if pd.notna(row.get('IAN')) and row['IAN'] <= 5.0:
        em_risco = True
    if pd.notna(row.get('Defasagem')) and row.get('Defasagem', 0) < 0:
        em_risco = True
    if pd.notna(row.get('Pedra')) and str(row.get('Pedra', '')).strip() in ['Quartzo', 'QUARTZO']:
        em_risco = True
    return 1 if em_risco else 0

df['em_risco'] = df.apply(criar_target, axis=1)
print(f"\nDistribuição do target:")
print(df['em_risco'].value_counts())
print(f"Proporção em risco: {df['em_risco'].mean()*100:.1f}%")

# Gráfico distribuição risco
fig, ax = plt.subplots(figsize=(8, 5))
counts = df['em_risco'].value_counts().sort_index()
bars = ax.bar(['Sem Risco', 'Em Risco'], counts.values, color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 10,
            f'{val} ({val/len(df)*100:.1f}%)', ha='center', fontweight='bold', fontsize=13)
ax.set_title('Distribuição: Alunos em Risco vs Sem Risco', fontweight='bold')
ax.set_ylabel('Número de Alunos')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '24_distribuicao_risco.png'), dpi=150, bbox_inches='tight')
plt.close()

# Features
feature_cols = ['IDA', 'IEG', 'IAA', 'IPS', 'IPP', 'IPV', 'INDE']
available_features = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 100]

# Features adicionais
df['Fase_num'] = pd.to_numeric(df.get('Fase'), errors='coerce')
if df['Fase_num'].notna().sum() > 100:
    available_features.append('Fase_num')

if 'Idade' in df.columns:
    df['Idade_num'] = pd.to_numeric(df['Idade'], errors='coerce')
    if df['Idade_num'].notna().sum() > 100:
        available_features.append('Idade_num')

if 'Ano_Ingresso' in df.columns:
    df['Anos_PM'] = df['Ano'] - pd.to_numeric(df['Ano_Ingresso'], errors='coerce')
    df['Anos_PM'] = df['Anos_PM'].clip(lower=0)
    if df['Anos_PM'].notna().sum() > 100:
        available_features.append('Anos_PM')

# Features derivadas
if 'IDA' in available_features and 'IEG' in available_features:
    df['IDA_x_IEG'] = df['IDA'] * df['IEG']
    available_features.append('IDA_x_IEG')
if 'IPS' in available_features and 'IAA' in available_features:
    df['IPS_x_IAA'] = df['IPS'] * df['IAA']
    available_features.append('IPS_x_IAA')

acad = [c for c in ['IDA', 'IEG', 'IAA'] if c in df.columns]
if len(acad) >= 2:
    df['Media_Academica'] = df[acad].mean(axis=1)
    available_features.append('Media_Academica')
psico = [c for c in ['IPS', 'IPP'] if c in df.columns]
if len(psico) >= 1:
    df['Media_Psico'] = df[psico].mean(axis=1)
    available_features.append('Media_Psico')

print(f"\nFeatures ({len(available_features)}): {available_features}")

# Preparar dados
df_model = df[available_features + ['em_risco']].dropna()
print(f"Dataset para modelagem: {df_model.shape}")

X = df_model[available_features]
y = df_model['em_risco']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")

# Escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modelos
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_weight = n_neg / n_pos if n_pos > 0 else 1

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_split=5,
        min_samples_leaf=2, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced', C=1.0
    ),
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        scale_pos_weight=scale_weight, random_state=42,
        eval_metric='logloss', use_label_encoder=False
    )
else:
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1, random_state=42
    )

# Treinar
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n--- {name} ---")

    if 'Logistic' in name:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train.values, X_test.values

    cv_scores = cross_val_score(model, X_tr, y_train, cv=cv, scoring='f1')
    print(f"  CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_proba': y_proba,
        'accuracy': acc, 'f1': f1, 'auc': auc,
        'cv_f1': cv_scores.mean(), 'cv_std': cv_scores.std()
    }

    print(f"  Acurácia: {acc:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print(f"  AUC-ROC:  {auc:.3f}")
    print(classification_report(y_test, y_pred, target_names=['Sem Risco', 'Em Risco']))

# Comparação
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

metrics_df = pd.DataFrame({
    name: {'Acurácia': r['accuracy'], 'F1-Score': r['f1'], 'AUC-ROC': r['auc']}
    for name, r in results.items()
}).T
metrics_df.plot(kind='bar', ax=axes[0], edgecolor='white', width=0.8)
axes[0].set_title('Comparação de Métricas', fontweight='bold')
axes[0].set_ylabel('Score'); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
axes[0].set_ylim(0, 1)

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})", linewidth=2)
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].set_title('Curvas ROC', fontweight='bold')
axes[1].set_xlabel('Falsos Positivos'); axes[1].set_ylabel('Verdadeiros Positivos')
axes[1].legend()

for name, r in results.items():
    prec, rec, _ = precision_recall_curve(y_test, r['y_proba'])
    axes[2].plot(rec, prec, label=name, linewidth=2)
axes[2].set_title('Precision-Recall', fontweight='bold')
axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '25_comparacao_modelos.png'), dpi=150, bbox_inches='tight')
plt.close()

# Matrizes de confusão
fig, axes_cm = plt.subplots(1, len(results), figsize=(6*len(results), 5))
if len(results) == 1:
    axes_cm = [axes_cm]
for i, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i],
                xticklabels=['Sem Risco', 'Em Risco'], yticklabels=['Sem Risco', 'Em Risco'])
    axes_cm[i].set_title(f'{name}', fontweight='bold')
    axes_cm[i].set_xlabel('Predito'); axes_cm[i].set_ylabel('Real')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, '26_matrizes_confusao.png'), dpi=150, bbox_inches='tight')
plt.close()

# Melhor modelo
best_name = max(results, key=lambda x: results[x]['auc'])
best_result = results[best_name]
best_model = best_result['model']
print(f"\nMelhor modelo: {best_name}")
print(f"  AUC-ROC: {best_result['auc']:.3f}")

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=available_features).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(importances)*0.4)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    importances.plot(kind='barh', ax=ax, color=colors)
    ax.set_title(f'Importância das Features - {best_name}', fontweight='bold')
    ax.set_xlabel('Importância Relativa')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, '27_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

# ========================================
# 4. SALVAMENTO DO MODELO
# ========================================
print("\n" + "=" * 60)
print("4. SALVAMENTO DO MODELO")
print("=" * 60)

# Retreinar com todos os dados
if 'Logistic' in best_name:
    X_all_scaled = scaler.fit_transform(X)
    best_model.fit(X_all_scaled, y)
else:
    best_model.fit(X.values, y)

joblib.dump(best_model, os.path.join(MODELS_DIR, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
joblib.dump(available_features, os.path.join(MODELS_DIR, 'feature_names.pkl'))

metadata = {
    'model_name': best_name,
    'features': available_features,
    'metrics': {
        'auc': float(best_result['auc']),
        'f1': float(best_result['f1']),
        'accuracy': float(best_result['accuracy'])
    },
    'target_description': 'em_risco: 1 = aluno em risco de defasagem, 0 = sem risco',
    'criteria': [
        'IAN <= 5.0 (defasagem moderada ou severa)',
        'Defasagem < 0',
        'Pedra = Quartzo'
    ]
}

with open(os.path.join(MODELS_DIR, 'model_metadata.json'), 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print("\nArtefatos salvos:")
for fname in os.listdir(MODELS_DIR):
    size = os.path.getsize(os.path.join(MODELS_DIR, fname))
    print(f"  {fname}: {size/1024:.1f} KB")

print("\nOutputs gerados:")
for fname in sorted(os.listdir(OUTPUTS_DIR)):
    print(f"  {fname}")

print("\n" + "=" * 60)
print("ANÁLISE COMPLETA!")
print("=" * 60)
