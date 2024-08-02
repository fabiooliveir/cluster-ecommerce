import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Clusterização de E-commerce",  # Define o título da aba do navegador
    layout="wide",                # Layout da página (pode ser 'centered' ou 'wide')
    initial_sidebar_state="expanded"  # Estado inicial da barra lateral (pode ser 'expanded' ou 'collapsed')
)

# Função para carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv', encoding='latin1')
    return df

# Função para pré-processar os dados
def preprocess_data(df):
    # Remover dados nulos
    df.dropna(subset=['CustomerID', 'Description'], inplace=True)

    # Remover transações com quantidade negativa ou zero
    df = df[df['Quantity'] > 0]

    # Remover transações com preço unitário negativo ou zero
    df = df[df['UnitPrice'] > 0]

    # Remover duplicados
    df.drop_duplicates(inplace=True)

    # Remover outliers (utilizando Z-score)
    df = df[(np.abs(stats.zscore(df[['Quantity', 'UnitPrice']])) < 3).all(axis=1)]

    # Criar variáveis agregadas para clusterização
    df['TotalSpent'] = df['Quantity'] * df['UnitPrice']
    customer_data = df.groupby('CustomerID').agg({
        'InvoiceNo': 'count',
        'TotalSpent': 'sum',
        'StockCode': pd.Series.nunique
    }).reset_index()

    customer_data.columns = ['CustomerID', 'PurchaseFrequency', 'TotalSpent', 'UniqueProducts']

    # Normalizar os dados
    scaler = MinMaxScaler()
    customer_data[['PurchaseFrequency', 'TotalSpent', 'UniqueProducts']] = scaler.fit_transform(
        customer_data[['PurchaseFrequency', 'TotalSpent', 'UniqueProducts']]
    )

    return customer_data

# Função para aplicar o algoritmo de clusterização
def apply_clustering(data, n_clusters):
    # Definindo o número de clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data[['PurchaseFrequency', 'TotalSpent', 'UniqueProducts']])

    # Análise dos clusters
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['PurchaseFrequency', 'TotalSpent', 'UniqueProducts'])
    return data, cluster_centers

# Função para plotar gráficos de clusters
def plot_clusters(data):
    st.write("### Visualização dos Clusters")

    # Visualização interativa usando Altair
    scatter_chart = alt.Chart(data).mark_circle(size=60).encode(
        x='PurchaseFrequency',
        y='TotalSpent',
        color='Cluster:N',
        tooltip=['CustomerID', 'PurchaseFrequency', 'TotalSpent', 'UniqueProducts', 'Cluster']
    ).interactive().properties(
        title='Clusters: Frequência de Compras vs. Gasto Total'
    )

    st.altair_chart(scatter_chart, use_container_width=True)

    scatter_chart_2 = alt.Chart(data).mark_circle(size=60).encode(
        x='UniqueProducts',
        y='TotalSpent',
        color='Cluster:N',
        tooltip=['CustomerID', 'PurchaseFrequency', 'TotalSpent', 'UniqueProducts', 'Cluster']
    ).interactive().properties(
        title='Clusters: Variedade de Produtos vs. Gasto Total'
    )

    st.altair_chart(scatter_chart_2, use_container_width=True)

    scatter_chart_3 = alt.Chart(data).mark_circle(size=60).encode(
        x='PurchaseFrequency',
        y='UniqueProducts',
        color='Cluster:N',
        tooltip=['CustomerID', 'PurchaseFrequency', 'TotalSpent', 'UniqueProducts', 'Cluster']
    ).interactive().properties(
        title='Clusters: Frequência de Compras vs. Variedade de Produtos'
    )

    st.altair_chart(scatter_chart_3, use_container_width=True)

# Visualização com PCA
def pca_visualization(data):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data[['PurchaseFrequency', 'TotalSpent', 'UniqueProducts']])

    data['PCA1'] = components[:, 0]
    data['PCA2'] = components[:, 1]

    pca_chart = alt.Chart(data).mark_circle(size=60).encode(
        x='PCA1',
        y='PCA2',
        color='Cluster:N',
        tooltip=['CustomerID', 'PurchaseFrequency', 'TotalSpent', 'UniqueProducts', 'Cluster']
    ).interactive().properties(
        title='Visualização em 2D dos Clusters com PCA'
    )

    st.altair_chart(pca_chart, use_container_width=True)

# Configuração do título e layout do aplicativo
st.title("Análise de Clusterização de Clientes de E-commerce")

# Seção de carregamento e exploração de dados
st.sidebar.header("Configurações de Análise")

# Carregar dados do arquivo CSV pré-definido
df = load_data()

st.sidebar.subheader("Opções de Clusterização")
n_clusters = st.sidebar.slider("Número de Clusters", 2, 10, 3)

# Visualização de dados
st.subheader("Distribuição das Variáveis")

# Distribuição de Países
st.write("#### Top 10 Países por Número de Transações")
top_countries = df['Country'].value_counts().head(10)
st.bar_chart(top_countries)

# Pré-processamento de dados
st.header("Pré-processamento de Dados")
customer_data = preprocess_data(df)

# Aplicação de clusterização
st.header("Clusterização de Clientes")
data, cluster_centers = apply_clustering(customer_data, n_clusters)
st.write("**Clusters Formados:**")
st.dataframe(data.groupby('Cluster').mean())

# Visualização dos clusters
plot_clusters(data)

# Visualização PCA
pca_visualization(data)

