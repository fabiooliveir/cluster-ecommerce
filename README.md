# Análise de Clusterização de Clientes de E-commerce

Este projeto realiza uma análise de clusterização de clientes de e-commerce para segmentar e entender melhor o comportamento dos clientes com base em suas transações. A aplicação é desenvolvida em Python utilizando Streamlit para visualização interativa dos dados e análise de clusterização.

## Funcionalidades

- **Carregamento e Exploração de Dados:** Visualize a distribuição das variáveis e principais informações dos dados.
- **Pré-processamento:** Limpeza e preparação dos dados, incluindo remoção de duplicados, tratamento de dados nulos e normalização.
- **Clusterização:** Aplicação do algoritmo K-Means para segmentar clientes em clusters com base em frequência de compras, gasto total e variedade de produtos.
- **Visualização:** Gráficos interativos mostrando a distribuição dos clusters e visualização em 2D utilizando PCA (Análise de Componentes Principais).
- **Interpretação dos Resultados:** Descrição dos clusters e sugestões para estratégias de marketing baseadas na análise.

## Instalação

1. Clone o repositório:

   ```bash
   git clone https://github.com/usuario/repo.git
   ```

2. Navegue para o diretório do projeto:

   ```bash
   cd repo
   ```

3. Crie um ambiente virtual (opcional, mas recomendado):

   ```bash
   python -m venv venv
   source venv/bin/activate  # No Windows use `venv\Scripts\activate`
   ```

4. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

5. Execute a aplicação Streamlit:

   ```bash
   streamlit run app.py
   ```

## Uso

- Após iniciar o Streamlit, acesse o aplicativo no navegador (normalmente em `http://localhost:8501`).
- Na barra lateral, ajuste o número de clusters desejado para ver diferentes segmentações dos clientes.
- Explore os gráficos e a interpretação dos clusters para entender o perfil dos clientes e sugerir ações de marketing.

## Licença

Este projeto está licenciado sob a Licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contribuição

Sinta-se à vontade para contribuir com o projeto criando issues, submetendo pull requests ou sugerindo melhorias. Para mais detalhes sobre como contribuir, consulte o arquivo [CONTRIBUTING.md](CONTRIBUTING.md).
```