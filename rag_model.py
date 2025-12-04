# rag_model.py
import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from openai import OpenAI

PERSIST_DIR = "./legal_chroma_db"
EMBEDDING_MODEL = "truro7/vn-law-embedding"

# Load DeepSeek API client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Load embedding + vectorstore
def load_vectorstore():
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

PROMPT_TEMPLATE = """
Bạn là Trợ lý Pháp lý AI chuyên phân tích hợp đồng theo Luật Việt Nam.

--- NGỮ CẢNH PHÁP LÝ (RAG) ---
{context}

--- NỘI DUNG HỢP ĐỒNG CẦN PHÂN TÍCH ---
{question}

--- YÊU CẦU ---
Hãy phân tích theo các mục:
1. Tóm tắt hợp đồng
2. Nghĩa vụ / quyền lợi quan trọng
3. Rủi ro pháp lý
4. Điều khoản bất lợi
5. Gợi ý chỉnh sửa
6. Đánh giá tuân thủ luật Việt Nam

Trả lời tiếng Việt, rõ ràng, mạch lạc.
"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

def generate_answer(context, question):
    final_prompt = prompt.format(context=context, question=question)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý pháp lý AI."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
        max_tokens=1024
    )

    return response.choices[0].message["content"]
