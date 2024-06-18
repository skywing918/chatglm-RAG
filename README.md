## 基于LangChain和ChatGLM3-6B的搜索引擎总结问答

> demo：调用本地ChatGLM3-6B模型，使用LangChain实现的RAG/搜索引擎agent

当前为CPU版本吗，如果需要更改以下代码：
api_server.py
```
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").float()
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
```

app-md-zilliz.py
```
embedding_function =  HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
embedding_function =  HuggingFaceBgeEmbeddings(
    model_name=EMBEDDING_PATH,
    model_kwargs={"device": "gpu"},
    encode_kwargs={"normalize_embeddings": True},
)
```

## 环境准备


1. 为防止本项目文件过大，项目中没有放模型文件，需要自己下载：

```bash
# 在根目录下创建model文件夹, 然后分别git clone ChatGLM模型和Embedding模型
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/AI-ModelScope/bge-large-zh-v1.5.git
```

目录结构是这样的：`./chatglm3-6b`和`./bge-large-zh-v1.5`


2. 创建、激活conda虚拟环境

```bash
$> conda create -n summary python=3.10
$> conda activate summary
```

3. 安装依赖

```bash
(summary) $> pip install -r requirements.txt
```

## 启动

1. 启动chatglm模型

```bash
(summary) $> python api_server.py
```

2. 然后新开一个终端测试

读取MD文件，上传到zilliz，结合bge-large-zh-v1.5和LLM,获取信息
需要申请zilliz，更改配置
```bash
(summary) $> python app-md-zilliz.py
```

已知信息作为prompt的一部分，通过LLM,获取信息
```bash
(summary) $> python promptApp.py
```
