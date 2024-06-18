from langchain_community.llms.chatglm3 import ChatGLM3
endpoint_url = "http://127.0.0.1:8000/v1/chat/completions"
model = ChatGLM3(
    endpoint_url=endpoint_url,
    temperature = 0,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

template = """<指令>根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题”，不允许在答案中添加编造成分，答案请使用中文。 </指令>\n
<已知信息>{context}</已知信息>\n
<问题>{question}</问题>\n"""
prompt = PromptTemplate.from_template(template)

chain = prompt | model | StrOutputParser()
context = "Apple有以下产品：IPhone15版：售价1万元;IPhone15 Pro版：售价2万元；"
question = "Apple有哪些产品，价格是多少？"

result = chain.invoke({"context": context, "question": question})
print(result)