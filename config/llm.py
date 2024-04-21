import os
from utils.utils import declare_env_vars

# For convenient purposes, user may declare environment variables in the code
# Note that system declared environment variables will take precedence
declare_env_vars("TOGETHER_API_KEY", None)
declare_env_vars("TOGETHER_API_BASE", None)

TOGETHER_API_URL = os.getenv('TOGETHER_API_BASE')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
TOGETHER_MODEL_ID = 'MISTRALAI/MISTRAL-7B-INSTRUCT-V0.2' # meta-llama/Llama-3-8b-chat-hf
USE_INST_TEMPLATE = True

DSPY_INPUT_DESC = "may contain relevant facts"
DSPY_OUTPUT_DESC = "leverages the context provided only to answer the question with no speculation, if the answer cannot be found in the text, please specify you dont have the knowledge to answer the question"