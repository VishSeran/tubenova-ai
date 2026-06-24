from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from modules.logger import get_logger
from langchain_classic.globals import set_verbose

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
        if len(summary) != 0 :
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
            logger.info("answer is fetched")
        
        return answer
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    
    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None
        
    
    
    
    