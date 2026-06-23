from youtube_transcript_api import YouTubeTranscriptApi
import re
from modules.logger import get_logger

logger = get_logger("data-extraction-logger")

# import video id 
def get_video_id (video_url:str):
    
    try:
        if not video_url:
            raise ValueError("Video url is empty or none")
        
        pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, video_url)
        
        return match.group(1) if match else None
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
    
    except Exception as e:
        logger.error(f"Error in get video id: {e}")
        
def transcript_process(video_url):
    
    try:    
        video_id = get_video_id(video_url)
        
        if video_id is None:
            logger.warning("Video id is None")
            return None
        
        yt_video_api = YouTubeTranscriptApi()
        fecthed_video =  yt_video_api.list(video_id)
        
        if not fecthed_video:
            raise ValueError("Video is not fetched")
        
        logger.info("Video is fetched!")
        
        transcript = ""
        
        for t in fecthed_video:
            if t.language_code=='en':
                if t.is_generated:
                    if len(transcript) == 0:
                        transcript = t.fetch()
                    
                else:
                    transcript = t.fetch()
        
        if not transcript:
            logger.warning("Transcript is not fetched")
            #raise ValueError("Transcript is not fetched")
            return None
        
        logger.info("Transcript has fetched")
        
        return transcript 
        
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        
    except Exception as e:
        logger.error(f"Error in video fetching: {e}")
    
    
def format_transcript(transcript):
    
    try:
        txt = ""
        
        if not transcript:
            raise ValueError("trainscript is empty or none")
        
        for t in transcript:
            txt += f"Text:{t.text} Start: {t.start}\n"
        
        if len(txt) == 0:
            logger.warning("Formatted transcript is empty and return None")
            return None    
        
        return txt
    
    except ValueError as e:
        logger.error(f"Value error: {e}")
        return None
    except Exception as e:
        logger.error(f"Error in fomartting the transcript: {e}")
        return None    
    
    