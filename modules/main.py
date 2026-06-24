from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from modules.logger import get_logger
from langchain_classic.globals import set_verbose

set_verbose(True)

logger = get_logger("main-logger")

def summary_generate(process_transcript, llm:ChatHuggingFace):
    
    try:
        if not process_transcript:
            raise ValueError("transcript is empty")
        
        template = """
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an AI assistant tasked with summarizing YouTube video transcripts
            Instructions:
            
            1. Summarize the transcript in a single concise paragraph.
            2. Ignore any timestamps in your summary.
            3. Focus on the spoken content (Text) of the video.
            
            Note: In the transcript, "Text" refers to the spoken words in the video,
            Please summarize the following YouTube video transcript:
            
            {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=['transcript']
        )
        
        summary_chain = prompt_template | llm | StrOutputParser()
        
        
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None     
    

def chat_with_llm(query, content, llm:ChatHuggingFace):
    
    try:
        
        if not query:
            raise ValueError("query is empty")
        
        if not content:
            raise ValueError("Video content is empty")
        
        template = """
        You are an expert assistant.

        Video Content:
        {content}

        Answer the following question using only the information provided in the video content.

        Question:
        {question}

        If the answer cannot be found in the video content, respond with:
        "I don't know based on the provided video content."
        """
        
        prompt_template = PromptTemplate(
            template=template,
            input_variables=['content', 'question']
            
        )
        
        chat_chain = prompt_template | llm | StrOutputParser() 
        logger.info("chat chain is created")
        
        answer = chat_chain.invoke(input={"content": content, "question": query})
        
        if len(answer) != 0 :
            logger.info("chat chain is created")
        
        return answer
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None
        
    
    
    
    