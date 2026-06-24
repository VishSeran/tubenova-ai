from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace
from modules.logger import get_logger

logger = get_logger("main-logger")


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
        
    
    
    
    