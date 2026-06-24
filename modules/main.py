from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from modules.logger import get_logger
from langchain_classic.globals import set_verbose
from modules.data_extraction import transcript_process, format_transcript
from modules.data_preprocess import txt_chunking
from modules.model_config import llm_model, create_vector_store, retrieve

set_verbose(True)

logger = get_logger("main-logger")

def summary_generate(process_transcript, llm:ChatHuggingFace):
    
    try:
        if not process_transcript:
            raise ValueError("transcript is empty")
        
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an AI assistant tasked with summarizing YouTube video transcripts.

            Summarize the transcript in one concise paragraph.
            Ignore timestamps and focus only on spoken content.

            Transcript:
            {transcript}
            """

        )
        
        summary_chain = prompt_template | llm | StrOutputParser()
        logger.info("Summaru chain is created")
        
        summary = summary_chain.invoke({"transcript": process_transcript})
        if summary:
            logger.info("answer is fetched")
        
        return summary
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None     
    

def chat_with_llm(query, content, llm:ChatHuggingFace):
    
    try:
        
        if not query:
            raise ValueError("Query is empty")
        
        if not content:
            raise ValueError("Video content is empty")
        

        prompt_template = ChatPromptTemplate.from_template(
            """
            You are an expert assistant.

            Video Content:
            {content}

            Answer the following question using only the information provided in the video content.

            Question:
            {question}

            If the answer cannot be found in the video content, respond with:
            "I don't know based on the provided video content."
        """
        )
        
        chat_chain = prompt_template | llm | StrOutputParser() 
        logger.info("chat chain is created")
        
        answer = chat_chain.invoke({"content": content, "question": query})
        
        if answer:
            logger.info("answer is fetched")
        
        return answer
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None
    

def main(video_url, query):
    
    try:
        if not query:
            raise ValueError("query is empty")

        transcript = transcript_process(video_url)
        logger.info("Transcript fetched")
        
        formatted_transcript = format_transcript(transcript)
        logger.info("Transcript formatted")
        
        chunks = txt_chunking(formatted_transcript)
        logger.info("Chunks fetched")
        
        vectorstore = create_vector_store(chunks)
        logger.info("Embedding model is initialized")
        logger.info("Vector Store is initialized")
        
        logger.info("Start retrieve process...")
        retrieve_results = retrieve(query, vectorstore)
        
        if not retrieve_results:
            logger.warning("retrieved results are empty or none")
            
        logger.info("results retrieved completed")
        
        
        
        
        
        
        
     
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None    
    
    
    
    