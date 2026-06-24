from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from modules.logger import get_logger
import os
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_classic.globals import set_verbose
from modules.data_extraction import get_video_id, transcript_process, format_transcript
from modules.data_preprocess import txt_chunking
from modules.model_config import (
    llm_model,
    create_vector_store,
    retrieve,
    embedding_model,
    save_index,
    load_vector_store,
)

set_verbose(True)

logger = get_logger("app-logger")
chat_llm = None
embed_model = None

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


logger.info("LLM model initializing...")
chat_llm = llm_model()

if not chat_llm:
    logger.warning("LLM model init failed")
else:
    logger.info("LLM model loaded successfull")


logger.info("Embedding model initializing...")

embed_model = embedding_model()

if not embed_model:
    logger.warning("Embedding model init failed")
else:
    logger.info("Embedding model loaded successfull")


def summary_generate_chain(process_transcript, llm: ChatHuggingFace = chat_llm):

    try:
        if not process_transcript:
            raise ValueError("transcript is empty")

        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a professional summarizer.

            Task:
            Summarize the given YouTube transcript into a clear, accurate paragraph.

            Rules:
            - Use ONLY the information in the transcript
            - Ignore timestamps, labels, and metadata
            - Do NOT add external knowledge
            - Do NOT repeat sentences
            - Keep it concise (5–7 sentences max)
            - Preserve key technical concepts and main idea

            Transcript:
            {transcript}
            """
        )

        summary_chain = prompt_template | llm | StrOutputParser()
        logger.info("Summary chain is created")

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


def chat_with_llm_chain(query, content, llm: ChatHuggingFace = chat_llm):

    try:
        if not query:
            raise ValueError("Query is empty")

        if not content:
            raise ValueError("Video content is empty")

        prompt_template = ChatPromptTemplate.from_template(
            
            """
            You must first find relevant sentences in the context.
            Then answer ONLY based on those sentences.

            If no relevant sentence exists → say "I don't know..."

            Context:
            {content}

            Question:
            {question}
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


def summary_generate(video_url):

    try:
        transcript = transcript_process(video_url)
        logger.info("Transcript fetched")

        formatted_transcript = format_transcript(transcript)
        logger.info("Transcript formatted")

        summary = summary_generate_chain(formatted_transcript)
        return summary

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None


def ingest_video(video_url):

    try:
        video_id = get_video_id(video_url)

        path = f"./data/faiss_index/{video_id}"

        if os.path.exists(path):
            logger.info("Index already exists")
            return "Already indexed"

        else:
            transcript = transcript_process(video_url)
            logger.info("Transcript fetched")

            formatted_transcript = format_transcript(transcript)
            logger.info("Transcript formatted")

            chunks = txt_chunking(formatted_transcript)
            logger.info("Chunks fetched")

            vectorstore = create_vector_store(chunks, embed_model)
            logger.info("Vector Store is initialized")

            save_index(vectorstore, video_id)
            return "Index built successfully"

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in saving vector store: {e}")
        return None


def ingest_and_flag(video_url):

    result = ingest_video(video_url)
    return result, True


def chat_with_llm(video_url, query, knowledge_ready):

    try:
        if not knowledge_ready:
            return "⚠ Please index the video first (Enable Knowledge)."

        else:
            video_id = get_video_id(video_url)
            loaded_vectorstore = load_vector_store(video_id, embed_model)

            if loaded_vectorstore is None:
                raise ValueError("Vector store loading failed")

            logger.info("Start retrieve process...")
            retrieve_results = retrieve(query, loaded_vectorstore)

            if not retrieve_results.strip():
                return "No relevant information found in video."

            logger.info("results retrieved completed")

            answer = chat_with_llm_chain(query, retrieve_results)
            return answer

    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None

    except Exception as e:
        logger.error(f"Error in retriever: {e}")
        return None
