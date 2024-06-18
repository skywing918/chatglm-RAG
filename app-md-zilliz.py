#pip install -qU langchain-text-splitters

from langchain_community.document_loaders import TextLoader

# Only keep post title, headers, and content from the full HTML.
loader = TextLoader("test.md", encoding="utf-8")
documents = loader.load()
#print(documents)
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header1"),
    ("##", "Header2"),
    ("###", "Header3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
all_splits = markdown_splitter.split_text(documents[0].page_content)

import os
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# set Embedding Model path
EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), './bge-large-zh-v1.5')
# create the open-source embedding function
embedding_function =  HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = Milvus.from_documents(
    all_splits,
    embedding = embedding_function,
    connection_args={
        "uri": "https://in03-d49fcef113295c7.api.gcp-us-west1.zillizcloud.com",
        "user": "",
        "password": "",
        # "token": ZILLIZ_CLOUD_API_KEY,  # API key, for serverless clusters which can be used as replacements for user and password
        "secure": True,
    },
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，"
    "不允许在答案中添加编造成分，答案请使用中文。 </指令>\n"
    "<已知信息>{context}</已知信息>\n"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

from langchain_community.llms.chatglm3 import ChatGLM3
endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
model = ChatGLM3(
    endpoint_url=endpoint_url,
    temperature = 0,
)


question_answer_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "OF内安装操作系统?"})
print(response["answer"])