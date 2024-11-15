from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


# Extract data from Pdf file
def load_pdf_file(data):
    loader= DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)

    documents=loader.load()
    return documents

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100,add_start_index=True,
                                                 strip_whitespace=True, separators=MARKDOWN_SEPARATORS)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


EMBEDDING_MODEL_NAME = "thenlper/gte-small"
def download_hugging_face_embeddings():
    embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)
    return embedding_model

