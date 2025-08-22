# RAG Service with LangChain + Qdrant

This project implements a RAG (Retrieval-Augmented Generation) service using LangChain, Qdrant, and FastAPI. The service can ingest documents, process them, and answer questions based on the ingested knowledge.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip
- Docker (for Ollama)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Muamm4/Rag-AITeam.git
    cd Rag-AITeam
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

To run the application, use the following command:

```bash
uvicorn app:app --reload
```

The application will be available at `http://localhost:8000`.

## Using Ollama with Docker

This project is configured to use Ollama for the LLM. You can run Ollama in a Docker container.

1.  **Pull the Ollama image:**

    ```bash
    docker pull ollama/ollama
    ```

2.  **Run the Ollama container:**

    ```bash
    docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
    ```

3.  **Pull the desired model:**

    ```bash
    docker exec -it ollama ollama pull gemma3:1b
    ```

## Switching LLM Providers

You can easily switch to other LLM providers like Gemini or OpenAI by modifying the `ingest.py` and `query.py` files.

### Using Google Gemini

1.  **Install the Google Generative AI SDK:**

    ```bash
    pip install langchain-google-genai
    ```

2.  **Set your Google API Key:**

    ```bash
    export GOOGLE_API_KEY="YOUR_API_KEY"
    ```

3.  **In `ingest.py` and `query.py`, change the `llm` and `embeddings` initialization:**

    ```python
    from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    llm = GoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    ```

### Using OpenAI

1.  **Install the OpenAI SDK:**

    ```bash
    pip install openai
    ```

2.  **Set your OpenAI API Key:**

    ```bash
    export OPENAI_API_KEY="YOUR_API_KEY"
    ```

3.  **In `ingest.py` and `query.py`, change the `llm` and `embeddings` initialization:**

    ```python
    from langchain_openai import OpenAI, OpenAIEmbeddings

    llm = OpenAI(model_name="gpt-3.5-turbo")
    embeddings = OpenAIEmbeddings()
    ```

## API Routes

The following API routes are available:

-   **`POST /ingest/`**: Ingests a document into the knowledge base.
    -   **Body**:
        -   `file`: An uploaded text file. (To implement)
        -   `text`: A string of text.
    -   **Returns**:
        -   `{"status": "ok", "chunks": <number-of-chunks>}` on success.
        -   `{"error": "Provide either file or text"}` if no data is provided.
        -   `{"error": "Error ingesting document"}` on failure.

-   **`POST /query/`**: Runs a query against the ingested knowledge.
    -   **Body**:
        -   `prompt`: The question to ask.
    -   **Returns**:
        -   `{"answer": <answer>}` on success.
        -   `{"detail": "Request timed out after 120 seconds."}` if the request times out.
        -   `{"detail": "An error occurred: <error-message>"}` on failure.

## Agents

The query process uses a chain of three agents to generate the final answer:

1.  **Intention Analyzer**: This agent reformulates the user's question to be clearer and more optimized for searching in a document database.

2.  **Document Retriever (RAG)**: This agent retrieves relevant documents from the knowledge base based on the optimized question. It uses a similarity search with a score threshold to find the most relevant document chunks.

3.  **Final Answer Generator**: This agent generates the final answer based on the retrieved documents and the original question. If the answer cannot be found in the documents, it will respond that there is not enough information to answer.