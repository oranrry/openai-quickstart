import gradio as gr
import random
import time
from typing import List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


def initialize_sales_bot(vector_store_dir: str="sale_script",index_name: str="real_estate"):
    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(openai_api_key="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"),index_name)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0,openai_api_key="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")
    
      

    bot = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    
    # 返回向量数据库的检索结果
    bot.return_source_documents = True

    return bot

def initialize_general_sales_bot():
    # 翻译任务指令始终由 System 角色承担
        template = (
            """You are the top {sales_scenario} salesperson in China. Please respond to the following questions based on {sales_scenario} sales pitches. \n
            The response needs to be concise, considering the user's emotions at the time."""
        )
        system_message_prompt = SystemMessagePromptTemplate.from_template(template)

        # 待翻译文本由 Human 角色输入
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
        chat_prompt_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        # 为了翻译结果的稳定性，将 temperature 设置为 0
        chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True,openai_api_key="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")
        global LLM   
        LLM= LLMChain(llm=chat, prompt=chat_prompt_template, verbose=True)
        return LLM

def sales_chat(message,history,sales_scenario):
    print(f"[message]{message}")
    print(f"[history]{history}")
    print(f"[sales_scenario]{sales_scenario}")
    # TODO: 从命令行参数中获取
    enable_chat = True
    global SALES_BOT

    if sales_scenario == 'real_estate':
        SALES_BOT = RE_SALES_BOT
    elif sales_scenario == 'appliances':
        SALES_BOT = APPLIANCES_SALES_BOT
    elif sales_scenario == 'home_decoration':
        SALES_BOT = HD_SALES_BOT
    elif sales_scenario == 'education_industry':
        SALES_BOT = EI_SALES_BOT
    else :
        SALES_BOT = RE_SALES_BOT


    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        result = ""
        try:
            result = LLM.run({
                "text": message,
                "sales_scenario": sales_scenario,
            })
        except Exception as e:
            print(f"An error occurred during translation: {e}")
            return result

        return result
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="销售咨询",
        # retry_btn=None,
        # undo_btn=None,
        additional_inputs=[gr.Dropdown(choices=["real_estate","appliances","home_decoration","education_industry"],value="real_estate",label="销售场景（默认家装行业）")],
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="0.0.0.0")

if __name__ == "__main__":
    global RE_SALES_BOT,APPLIANCES_SALES_BOT,HD_SALES_BOT,EI_SALES_BOT
    # 初始化房产销售机器人
    RE_SALES_BOT=initialize_sales_bot(index_name="real_estate")
    APPLIANCES_SALES_BOT=initialize_sales_bot(index_name="appliances")
    HD_SALES_BOT=initialize_sales_bot(index_name="home_decoration")
    EI_SALES_BOT=initialize_sales_bot(index_name="education_industry")
    # 启动 Gradio 服务
    launch_gradio()
