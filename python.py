from groq import Groq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# List models
models = client.models.list()
print(models)
