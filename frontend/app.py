import streamlit as st
import requests


BACKEND_URL = 'http://backend:8000/query'

st.title("POC chatbot to Huma.AI assessment")


query = st.text_input("type your question:")


model = st.selectbox(
    "Select the model:",
    ('gpt', 'claude', 'gemini')
)

if st.button("Sent"):
    if query.strip() == "":
        st.warning("Please, do a question.")
    else:

        payload = {
            'query': query,
            'model': model
        }

        try:

            response = requests.post(BACKEND_URL, json=payload)
            response.raise_for_status() 

            data = response.json()
            answer = data.get('answer', 'No answer received.')


            st.markdown("### Answer:")
            st.write(answer)

        except requests.exceptions.RequestException as e:
            st.error(f"Error: {e}")
