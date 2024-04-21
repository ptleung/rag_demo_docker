import os
from utils.utils import declare_env_vars

# For convenient purposes, user may declare environment variables in the code
# Note that system declared environment variables will take precedence
declare_env_vars("LLAMAPARSE_API_KEY", None) 
declare_env_vars("HUGGINGFACE_API_KEY", None)
declare_env_vars("REDIS_ACCT", None)
declare_env_vars("REDIS_PW", None)

LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

DATA_FILE_PATH = "./data/parsed_data.pkl"

PARSING_INSTRUCTION = """The provided documents are different insurance products offered by Prudential HK.
        The documents included some high level introduction to the insurance products.
        It includes insurance plan highlights, benefits, how the plan works, details of the plan, and risks. 
        The content may vary as the product introduction is different for each insurance product.
        It may contain tables in the document.
        Try to be precise when extracting the content from the document."""

HUGGINGFACE_MODEL_ID = "WhereIsAI/UAE-Large-V1"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_EMBEDDING_MODEL_URL = f"https://api-inference.huggingface.co/models/{HUGGINGFACE_MODEL_ID}"
HUGGINGFACE_EMBEDDING_API_HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

REDIS_ACCT = os.getenv("REDIS_ACCT")
REDIS_PW = os.getenv("REDIS_PW")
REDIS_HOST = 'redis-10852.c56.east-us.azure.redns.redis-cloud.com'
REDIS_PORT = 10852  
REDIS_URL = f"redis://{REDIS_ACCT}:{REDIS_PW}@{REDIS_HOST}"
REDIS_INDEX_NAME = "pru_docs"

# VECTOR_DIMENSIONS = 1024
NUM_DOCS_RETURNED = 3