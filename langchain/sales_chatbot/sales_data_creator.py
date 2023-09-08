from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def data_create(data_path: str="real_estate_sales_data.txt"):
    with open(data_path) as f:
        real_estate_sales = f.read()

    text_splitter = CharacterTextSplitter(        
        separator = r'\d+\.',
        chunk_size = 100,
        chunk_overlap  = 0,
        length_function = len,
        is_separator_regex = True,
    )

    docs = text_splitter.create_documents([real_estate_sales])
    db = FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key="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"))
   # db.save_local("sale_script",index_name="real_estate")
   # db.save_local("sale_script",index_name="appliances")
   # db.save_local("sale_script",index_name="home_decoration")
   # db.save_local("sale_script",index_name="education_industry")


#data_create(data_path="/Users/yiqing/AIGC Class/jupyter lab/openai-quickstart/langchain/sales_chatbot/real_estate_sales_data.txt")
#data_create(data_path="/Users/yiqing/AIGC Class/jupyter lab/openai-quickstart/langchain/sales_chatbot/appliances_sales_data.txt")
#data_create(data_path="/Users/yiqing/AIGC Class/jupyter lab/openai-quickstart/langchain/sales_chatbot/home_decoration_sales_data.txt")
#data_create(data_path="/Users/yiqing/AIGC Class/jupyter lab/openai-quickstart/langchain/sales_chatbot/education_industry_sales_data.txt")