from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
HUGGINGFACEHUB_API_TOKEN = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

embeddings = download_hugging_face_embeddings()

index_name = 'medichatbot2'
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 5})

repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=500,
    temperature=0.3,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['GET','POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = qa_chain({"query": msg})
    print("Response: ", response['result'])
    return str(response['result'])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)