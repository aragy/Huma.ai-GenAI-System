import streamlit as st
import requests

# Defina a URL da API backend
BACKEND_URL = 'http://backend:8000/query'

st.title("POC chatbot to Huma.AI assessment")

# Entrada para a consulta do usuário
query = st.text_input("type your question:")

# Seleção do modelo
model = st.selectbox(
    "Select the model:",
    ('gpt', 'claude', 'gemini')
)

if st.button("Sent"):
    if query.strip() == "":
        st.warning("Please, do a question.")
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
            answer = data.get('answer', 'No answer received.')

            # Exiba a resposta
            st.markdown("### Answer:")
            st.write(answer)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
