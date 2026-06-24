from modules.logger import get_logger
from dotenv import load_dotenv
import os
from modules.config import MODEL_NAME,EMBED_MODEL_NAME
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
import faiss
from huggingface_hub import login

logger = get_logger("model-config-logger")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def llm_model(model_name=MODEL_NAME):
    try:
        
        if not model_name:
            raise ValueError("model name is empty or none")
        
        login(HF_TOKEN)
        
        pipeline_kwargs = {"max_new_tokens": 256}
        
        pipeline = HuggingFacePipeline.from_model_id (
            model_id=model_name,
            pipeline_kwargs=pipeline_kwargs,
            task="text-generation",
            model_kwargs= {
                "token": HF_TOKEN
            },

        )
        
        llm = ChatHuggingFace (llm=pipeline, temperature=0.9)
        if not llm:
            logger.warning("LLM error")
        
        logger.info(f"{model_name} launched successful")
        
        return llm 
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in model config: {e}")
        return None
    
def embedding_model (model_name=EMBED_MODEL_NAME):
    
    try:
        if not model_name:
            raise ValueError("embedding model name is empty or none")
        
        embed_model = HuggingFaceEmbeddings(
            model_name = model_name,
            
        )
    
        logger.info(f"{model_name} launched successfull")
        return embed_model
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in embedding model: {e}")
        return None
    
    
def create_vector_store(chunks, embedding):
    
    try:
        
        if not chunks:
            raise ValueError("Chunks are empty")
        
        dimension = len(embedding.embed_query("test"))
        
        if not dimension or dimension == 0:
            raise ValueError(f"dimension value is: {dimension}")

        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        
        docstore = InMemoryDocstore({})
        
        vectorstore = FAISS(
            embedding_function=embedding,
            index=index,
            docstore=docstore,
            index_to_docstore_id={}
        )
        
        docs = [Document(page_content=chunk) for chunk in chunks]
        vectorstore.add_documents(docs)
        
        return vectorstore
    

        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in create vector index: {e}")
        return None
    
    
    
def save_index(vectorstore, video_id):
    
    try:
        
        if not video_id:
            raise ValueError("Video id is not found")
        
        path = f"./data/faiss_index/{video_id}"
        os.makedirs(path, exist_ok=True)
        vectorstore.save_local(path)
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in create vector index: {e}")
        return None     

def load_vector_store(video_id, embedding):
    
    path = f"./data/faiss_index/{video_id}"
    try:
        if not video_id:
            raise ValueError("Video id si not found")
        
        if embedding is None:
            raise ValueError("Embedding model not loaded")
        
        vector_store = FAISS.load_local(
            path,
            embedding,
            allow_dangerous_deserialization=True
        )
        
        logger.info("Index loaded successfull")
        return vector_store

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in create vector index: {e}")
        return None
    
def retrieve(query, vectorstore, k=4):
    
    try:
        
        if not query:
            raise ValueError("Query cannot be empty or none")
        
        if not vectorstore:
            raise ValueError("VectorStore cannot be empty or none")
    
        retriever = vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k":k,"fetch_k": 10}
        )
        
        results = retriever.invoke(query)
        text = "\n".join([doc.page_content for doc in results])
        return text
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None
    


