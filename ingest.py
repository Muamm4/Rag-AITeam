from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import getpass
import uuid
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API key: ")

OLLAMA_HOST = "http://localhost:11434"
QDRANT_COLLECTION = "rag_docs"

client = QdrantClient(host="localhost", port=6333)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = OllamaLLM(model="gemma3:1b", temperature=0.2, base_url=OLLAMA_HOST)

metadata_prompt = PromptTemplate(
    template="""Para o seguinte trecho de texto, extraia um resumo conciso e palavras-chave relevantes.
    Palavras chaves não devem serem repetidas, e devem ser relevantes para o trecho de texto, sempre trazer com letras minúsculas.
    Texto: {chunk}
    Formato da Saída:
    Resumo: <resumo>
    Palavras-chave: <palavra1>, <palavra2>, <palavra3>
    """,
    input_variables=["chunk"],
)


metadata_chain = {"chunk": RunnablePassthrough()} | metadata_prompt | llm

def ingest_document(content: str):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        
        metadata_list = []
        doc_id = str(uuid.uuid4())
        
        # Gera metadados para cada pedaço de texto usando o LLM
        for i, chunk in enumerate(chunks):
            response = metadata_chain.invoke({"chunk": chunk})
            print(response)
            
            summary = "Sem resumo"
            keywords = []
            
            lines = response.strip().split('\n')
            for line in lines:
                if line.startswith("Resumo:"):
                    summary = line.replace("Resumo:", "").strip()
                elif line.startswith("Palavras-chave:"):
                    keywords = [k.strip() for k in line.replace("Palavras-chave:", "").split(',')]
            
            metadata_list.append({
                "doc_id": doc_id,
                "sequence_number": i,
                "source": "LLM_generated",
                "summary": summary,
                "keywords": keywords,
            })
            
        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION,
            embedding=embeddings,
        )
        
        vectorstore.add_texts(chunks, metadatas=metadata_list)
        print(f"Ingestão concluída. {len(chunks)} chunks adicionados com metadados.")
        return len(chunks)
        
    except Exception as e:
        print(f"Erro na ingestão do documento: {e}")
        return 0