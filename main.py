import nest_asyncio

from llama_index.core import SummaryIndex
from llama_index.core import KnowledgeGraphIndex
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings, SimpleDirectoryReader, PromptTemplate
from llama_index.core import StorageContext, ServiceContext
from llama_index.core.prompts import RichPromptTemplate

from llama_index.graph_stores.neo4j import Neo4jGraphStore

from llama_index.llms.ollama import Ollama

#from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding

## to use Qdrant, install it w/ poetry add llama-index-vector-stores-qdrant
## note: vectorstore not supporting python 3.13 yet, 
#from llama_index.vector_stores.qdrant import QdrantVectorStore
#import qdrant_client

from llama_index.vector_stores.redis import RedisVectorStore
from redis import Redis

from config import config

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nest_asyncio.apply()


def create_qdrant_index(documents):
    client = qdrant_client.QdrantClient(
        host="localhost",
        port=6333
    )
    vector_store = QdrantVectorStore(client=client, collection_name="qdrant_collection")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    return index

def create_redis_index(documents):
    redis_client = Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        password=config.REDIS_PASSWORD,
        username=config.REDIS_USERNAME,
        db=0
    )
    logger.info("Redis client created: %s", redis_client.ping())

    # create the vector store wrapper
    vector_store = RedisVectorStore(redis_client=redis_client, overwrite=True)

    # load storage context
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # build and load index from documents and storage context
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    # index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return index


# # setup llm & embedding model
# llm=Ollama(model=config.OLLAMA_LLM_MODEL, base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}", 
#            request_timeout=600.0)
# Settings.llm = llm
# # embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
# embed_model = OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL, 
#                               base_url=f"http://{config.OLLAMA_HOST}:{config.OLLAMA_PORT}", 
#                               trust_remote_code=True)

# setup llm & embedding model
llm=Ollama(model=config.OLLAMA_LLM_MODEL, base_url=f"http://{config.OLLAMA_HOST}", 
           request_timeout=600.0)
Settings.llm = llm
# embed_model = HuggingFaceEmbedding( model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
embed_model = OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL, 
                              base_url=f"http://{config.OLLAMA_HOST}", 
                              trust_remote_code=True)

Settings.embed_model = embed_model

# load data
loader = SimpleDirectoryReader(
            input_dir = config.DOC_DIR,
            required_exts=[".pdf"],
            recursive=True
        )
docs = loader.load_data()

# # Creating a vector index over loaded data
# logger.info('Creating vector index')
# try:
#     index = create_redis_index(docs)
#     logger.info('Using Redis collection')
# except:
#     logger.warning("Failed to create Redis collection, using local collection")
#     index = VectorStoreIndex.from_documents(docs, show_progress=True)


# # ====== Customise prompt template ======
# qa_prompt_tmpl_str = (
# "Context information is below.\n"
# "---------------------\n"
# "{context_str}\n"
# "---------------------\n"
# "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
# "Query: {query_str}\n"
# "Answer: "
# )
# qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

# query_engine.update_prompts(
#     {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
# )
# response = query_engine.query(user_query,)


# Create the knowledge graph index
uri = config.NEO4J_URI
username = config.NEO4J_USERNAME
password = config.NEO4J_PASSWORD

graph_store = Neo4jGraphStore(
    username=username,
    password=password,
    url=uri,
    database="neo4j",
    timeout=600.0
)

storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: only need once to build the KG, can take a while!
# kg_index = KnowledgeGraphIndex.from_documents(
#     docs,
#     storage_context=storage_context,
#     max_triplets_per_chunk=8,
#     embed_model=embed_model,
#     show_progress=True
# )

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever

graph_rag_retriever = KnowledgeGraphRAGRetriever(
    storage_context=storage_context,
    verbose=True,
)

query_engine = RetrieverQueryEngine.from_args(
    graph_rag_retriever,
    embed_model=embed_model,
)

#-----------------------------
#########Example from the site#########(No improvements)
# text_qa_template_str = """Context information is below:
# <context>
# {{ context_str }}
# </context>

# Using both the context information and also using your own knowledge, answer the question:
# {{ query_str }}
# """
# text_qa_template = RichPromptTemplate(text_qa_template_str)

# refine_template_str = """New context information has been provided:
# <context>
# {{ context_msg }}
# </context>

# We also have an existing answer generated using previous context:
# <existing_answer>
# {{ existing_answer }}
# </existing_answer>

# Using the new context, either update the existing answer, or repeat it if the new context is not relevant, when answering this query:
# {query_str}
# """


# refine_template = RichPromptTemplate(refine_template_str)
    
# query_engine.update_prompts(
#     {
#         "response_synthesizer:text_qa_template": text_qa_template,
#         "response_synthesizer:refine_template": refine_template,
#     }
# )


####### Customise prompt template #######
prompt1 = RichPromptTemplate("""You are a graphDB query assistant. Your task is to:  
            1. Extract the core keywords/phrases from the user's unstructured question.  
            2. For each keyword, fetch related nodes and edges from the graphDB (e.g., synonyms, hypernyms, or closely linked concepts).  
            3. Return a structured summary of the keyword context to inform query rewriting.  

            **User Question:**  
            {{ user_query }}  

            **Output Format:**  
            - Extracted Keywords: [List of keywords]  
            - Retrieved Context for Each Keyword:  
            - "[Keyword 1]": [Related nodes/edges from graphDB]  
            - "[Keyword 2]": [Related nodes/edges from graphDB]  
            - ...  """)

prompt2 = RichPromptTemplate("""Using the retrieved keyword context below, answer the user's original question

            **User Question:**
            {{ user_query }}
            
            **Retrieved Keyword Context:**
            {{ keyword_context }}
            """)
#-----------------------------


# get user query
while (user_query := input("\n\nWhat do you want to know about these files?\n")):
    query1 = prompt1.format(user_query=user_query)
    response1 = query_engine.query(query1)
    
    keyword_context = str(response1).split("</think>")[1]
    
    query2 = prompt2.format(user_query=user_query, keyword_context=keyword_context)
    response2 = query_engine.query(query2,)

    print(str(response2))
