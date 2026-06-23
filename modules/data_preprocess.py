from modules.logger import get_logger
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = get_logger("data-preprocess-logger")

def txt_chunking(texts, chunk_size=500, chunk_overlap = 50):
    
    try:
        if not texts or not texts.strip():
            raise ValueError ("Texts are empty or none")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info("Text splitter created")
        
        split_texts = text_splitter.split_text(texts)
        if not split_texts:
            raise ValueError("error is split texts")
        logger.info("Texts are splitted successful")
        
        return split_texts
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in text splitting: {e}")
        return None
    
