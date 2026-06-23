from youtube_transcript_api import YouTubeTranscriptApi
import re

# import video id 
def get_video_id (video_url:str):
    
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, video_url)
    
    return match.group(1) if match else None
    
    
    
    