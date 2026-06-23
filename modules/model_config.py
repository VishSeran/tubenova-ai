from modules.logger import get_logger
from dotenv import load_dotenv
import os
import requests
from modules.config import MODEL_NAME
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

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

