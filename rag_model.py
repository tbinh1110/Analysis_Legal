import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

# Path lưu vectorstore đã train
PERSIST_DIR = "/tmp/legal_chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# Load DeepSeek API client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Load vectorstore đã train sẵn (chỉ load từ persist)
vectorstore = Chroma(persist_directory=PERSIST_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt template
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
        max_tokens=512
    )

    return response.choices[0].message["content"]
