import streamlit as st
import requests

# Defina a URL da API backend
BACKEND_URL = 'http://backend:8000/query'

st.title("Chatbot com Recuperação Aumentada por Geração (RAG)")

# Entrada para a consulta do usuário
query = st.text_input("Digite sua consulta:")

# Seleção do modelo
model = st.selectbox(
    "Selecione o modelo para processar sua consulta:",
    ('gpt', 'claude', 'gemini')
)

if st.button("Enviar"):
    if query.strip() == "":
        st.warning("Por favor, insira uma consulta.")
    else:
        # Prepare os dados para enviar
        payload = {
            'query': query,
            'model': model
        }

        try:
            # Envie a solicitação para a API backend
            response = requests.post(BACKEND_URL, json=payload)
            response.raise_for_status()  # Levanta uma exceção para erros HTTP

            data = response.json()
            answer = data.get('answer', 'Nenhuma resposta recebida.')

            # Exiba a resposta
            st.markdown("### Resposta:")
            st.write(answer)

        except requests.exceptions.RequestException as e:
            st.error(f"Ocorreu um erro: {e}")
