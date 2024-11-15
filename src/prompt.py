from langchain import hub

# Loads the latest version
prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")