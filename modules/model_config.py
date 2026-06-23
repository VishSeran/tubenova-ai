from modules.logger import get_logger
from dotenv import load_dotenv
import os
import requests
from modules.config import MODEL_NAME

logger = get_logger("model-config-logger")

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


