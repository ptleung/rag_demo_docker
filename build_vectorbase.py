import joblib
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Redis
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

from config.vectordb import LLAMAPARSE_API_KEY, DATA_FILE_PATH, PARSING_INSTRUCTION, HUGGINGFACE_MODEL_ID, HUGGINGFACE_API_KEY, REDIS_ACCT, REDIS_PW, REDIS_HOST, REDIS_PORT, REDIS_INDEX_NAME

def load_or_parse_data():
    """Parse the documents using LlamaParse with instructions and store the parsed data in a file."""
    if os.path.exists(DATA_FILE_PATH):
        # Load the parsed data from the file
        parsed_data = joblib.load(DATA_FILE_PATH)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parser = LlamaParse(api_key=LLAMAPARSE_API_KEY,
                            result_type="markdown",
                            parsing_instruction=PARSING_INSTRUCTION,
                            max_timeout=5000,)
        file_extractor = {".pdf": parser}
        parsed_data = SimpleDirectoryReader(
            "./docs", file_extractor=file_extractor
        ).load_data()

        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(parsed_data, DATA_FILE_PATH)
    return parsed_data

def add_source_doc_in_chunk(doc):
    """Add source document name in the chunk to make sure the source document is known for each chunk."""
    doc_name = doc.metadata['source']
    formatted_doc_name = doc_name.replace('data/pruhealth', "Prudential's").replace('.md', '').replace('-', ' ')
    doc.page_content = "Source Document: " + formatted_doc_name + "\n" + doc.page_content
    return doc

# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()

    # Find the file name in llama_parse_documents[i].metadata['file_name']
    for doc in llama_parse_documents:
        # Write the doc.text into the file
        with open(os.path.join('data', doc.metadata['file_name'].replace('pdf', 'md')), 'w+') as f:
            f.write(doc.text + '\n')

    markdown_path = "data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

    loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()
    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    docs = text_splitter.split_documents(documents)

    # Add source document name in the chunk to make sure the source document is known for each chunk
    docs = [add_source_doc_in_chunk(doc) for doc in docs]

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")

    # Initialize Embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HUGGINGFACE_API_KEY, model_name=HUGGINGFACE_MODEL_ID
    )

    url = f"redis://{REDIS_ACCT}:{REDIS_PW}@{REDIS_HOST}:{REDIS_PORT}"

    vs = Redis.from_documents(
        docs, embeddings, redis_url=url, index_name=REDIS_INDEX_NAME
    )
    vs.write_schema('redis_schema.yaml')

    print('Vector DB created successfully !')

if __name__ == "__main__":
    create_vector_database()