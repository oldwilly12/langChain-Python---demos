import os
from dotenv import load_dotenv

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

