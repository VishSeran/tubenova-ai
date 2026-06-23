from modules.logger import get_logger
from dotenv import load_dotenv
import os
from modules.config import MODEL_NAME,EMBED_MODEL_NAME
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document

logger = get_logger("model-config-logger")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def llm_model(model_name=MODEL_NAME):
    try:
        
        if not model_name:
            raise ValueError("model name is empty or none")
        
        pipeline_kwargs = {"max _new _tokens": 256}
        
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
    
embedding = embedding_model()
dimension = len(embedding.embed_query("test"))


def create_vector_store(chunks, embedding_model):
    
    try:
        
        if not dimension or dimension == 0:
            raise ValueError(f"dimension value is: {dimension}")

        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 50
        
        docstore = InMemoryDocstore({})
        
        vectorstore = FAISS(
            embedding_function=embedding_model,
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
    
def retrieve(query, vectorstore, k=4):
    
    try:
    
        retriever = vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k":k}
        )
        results = retriever.invoke(query)
        return results
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None