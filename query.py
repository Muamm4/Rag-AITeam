from langchain_ollama import OllamaLLM 
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import os
import getpass
load_dotenv()


if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

QDRANT_URL = "http://localhost:6333"
OLLAMA_HOST = "http://localhost:11434"
QDRANT_COLLECTION = "rag_docs"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = OllamaLLM(model="gemma3:1b", temperature=0.2, base_url=OLLAMA_HOST)
client = QdrantClient(host="localhost", port=6333)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=QDRANT_COLLECTION,
    embedding=embeddings,
)

intention_prompt = PromptTemplate(
    template="""Você é um analista de intenção. Sua tarefa é reformular a seguinte pergunta para torná-la
    mais clara e otimizada para uma busca em uma base de dados de documentos. Mantenha o foco principal.

    Contexto:
    Pergunta Original: {question}

    Como saida deve ser Somente a pergunta otimizada, exemplos:
    "Definição do Laravel"
    "Documentação do Laravel"
    """,
    input_variables=["question"],
)

intention_chain = {"question": RunnablePassthrough()} | intention_prompt | llm

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.85}
)

final_answer_prompt = PromptTemplate(
    template="""Responda a pergunta apenas com base nas informações fornecidas nos documentos recuperados abaixo.
    Se a resposta não puder ser encontrada nos documentos, responda "Não há informações suficientes nos documentos para responder a esta pergunta."
    Não adicione informações que não estejam nos documentos.

    Pergunta: {question}

    Documentos recuperados:
    {context}

    Resposta:
    """,
    input_variables=["context", "question"],
)

final_answer_chain = {"question": RunnablePassthrough(), "context": RunnablePassthrough()} | final_answer_prompt | llm

def reorder_retrieved_chunks(retrieved_docs: list[dict]) -> list[dict]:
    grouped_chunks = {}
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id not in grouped_chunks:
            grouped_chunks[doc_id] = []
        grouped_chunks[doc_id].append(doc)
    
    ordered_docs = []
    for doc_id, chunks_list in grouped_chunks.items():
        sorted_chunks = sorted(chunks_list, key=lambda x: x.metadata.get("sequence_number", 0))
        ordered_docs.extend(sorted_chunks)
        
    return ordered_docs



def query_document(question: str):
    """
    Executa a consulta com uma cadeia de 3 agentes.
    """
    try:
        optimized_query_response = intention_chain.invoke({"question": question})
        optimized_query = optimized_query_response.strip()
        
        rag_response = retriever.invoke(f"{optimized_query}")
        doc_snippet = None
        if rag_response:
            doc_ordered = reorder_retrieved_chunks(rag_response)
            doc_snippet = "\n\n".join([doc.page_content for doc in doc_ordered])
        else:
            doc_snippet = "Nenhum documento relevante encontrado."
        
        final_answer_response = final_answer_chain.invoke({
            "context": doc_snippet,
            "question": question,
        })
        
        final_answer = final_answer_response.strip()
        
        return final_answer
        
    except Exception as e:
        return "Desculpe, não foi possível processar sua solicitação."