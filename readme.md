# Huma.AI - RAG Chatbot POC 

This project implements a Retrieval-Augmented Generation (RAG) chatbot as part of a test for Huma.ai. It allows users to interact with advanced language models (GPT, Claude, and Gemini) to process queries. The system consists of a backend served by FastAPI and a frontend built with Streamlit for user interaction.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Environment Variables](#environment-variables)


## Overview

The system allows users to:

- Input queries in natural language.
- Choose among three language models to process the query:
  - **GPT** (OpenAI)
  - **Claude** (Anthropic)
  - **Gemini** (Google Generative AI)
- Receive generated responses based on retrieved documents and web information.
- View the sources used in generating the response.

This project is designed to showcase the capabilities of combining retrieval mechanisms with advanced language models to provide informative and accurate answers.

## Project Structure
```bash
Huma.ai-GenAI-System/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── models.py
│   │   ├── retriever.py
│   │   ├── evaluator.py
│   │   ├── knowledge_refiner.py
│   │   ├── query_rewriter.py
│   │   ├── web_searcher.py
│   │   ├── response_generator.py
│   │   ├── query_processor.py
│   │   └── utils.py
│   ├── data/
│   │   └── cleaned_dataset.csv
│   ├── .env
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── README.md

```

## Prerequisites

- **Docker** installed on your system.
- **Docker Compose** installed.

## Installation

Follow these steps to set up the project on your local machine.

### 1. **Clone the Repository**

Clone the project repository from GitHub to your local machine:

```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository

```

### 2. **Set Up Environment Variables**

The backend requires API keys to access the language models (GPT, Claude, and Gemini). These API keys are essential for the application to communicate with the respective services. Follow the steps below to securely set up your environment variables.


1. Create the `.env` File:

    Create a file named `.env` in the `backend/` directory. 
    ```bash

    touch .env
    ```


2. Add Your API Keys to the `.env` File:

  Open the .env file in your preferred text editor and add the following lines, replacing the placeholders with your actual API keys:

  ```bash
  OPENAI_API_KEY=your_openai_api_key
  ANTHROPIC_API_KEY=your_anthropic_api_key
  GOOGLE_GENAI_API_KEY=your_google_genai_api_key
  ```

## Running the Application
At the root of the project, run the following command to build and start the containers:

```bash
docker-compose up --build
```
---


## Usage

1. **Access the User Interface:**

   Open your web browser and navigate to [http://localhost:8501](http://localhost:8501).

2. **Interact with the Chatbot:**

   - Enter your query in the text input field.
   - Select the language model you wish to use (GPT, Claude, or Gemini) from the dropdown menu.
   - Click the "Submit" button to receive a response.

3. **View the Response:**

   The generated response will be displayed on the screen, including any sources used.


## Environment Variables

The system uses the following environment variables:

- **Backend (`backend/.env`):**

  - `OPENAI_API_KEY`: API key for OpenAI GPT.
  - `ANTHROPIC_API_KEY`: API key for Anthropic Claude.
  - `GOOGLE_GENAI_API_KEY`: API key for Google Generative AI Gemini.

- **Frontend:**

  - Currently, the frontend does not require additional environment variables.

**Note:** API keys are required to access the language models and must be obtained directly from the respective providers.
