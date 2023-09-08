import gradio as gr
import faiss
from langchain.docstore import InMemoryDocstore
from typing import List

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

from langchain_experimental.autonomous_agents import AutoGPT
from langchain.chat_models import ChatOpenAI

import os

os.environ["SERPAPI_API_KEY"] = "e65622355785aba531fe0f3733c6c429e3ec43457c916a0c3006e6f81d433369"
def initialize_auto_gpt():
    # 构造 AutoGPT 的工具集
    search = SerpAPIWrapper()
    tools = [
        Tool(
            name="search",
            func=search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        ),
        WriteFileTool(),
        ReadFileTool(),
    ]

    # OpenAI Embedding 模型
    embeddings_model = OpenAIEmbeddings(model="gpt-3.5-turbo", temperature=0,openai_api_key="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")
    # OpenAI Embedding 向量维数
    embedding_size = 1536
    # 使用 Faiss 的 IndexFlatL2 索引
    index = faiss.IndexFlatL2(embedding_size)
    # 实例化 Faiss 向量数据库
    vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

    global AGENT
    AGENT= AutoGPT.from_llm_and_tools(
        ai_name="Jarvis",
        ai_role="Assistant",
        tools=tools,
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0,openai_api_key="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"),
        memory=vectorstore.as_retriever(), # 实例化 Faiss 的 VectorStoreRetriever
    )

    # 打印 Auto-GPT 内部的 chain 日志
    AGENT.chain.verbose = True

def work(message,history):
    goals=[message]
    AGENT.run(goals)
    message = "".join([msg.content for msg in AGENT.chat_history_memory.messages])
    return message

def launch_gradio():
    demo = gr.ChatInterface(
        fn=work,
        title="目标描述",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")



if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_auto_gpt()
    # 启动 Gradio 服务
    launch_gradio()



