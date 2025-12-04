import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

# 1️⃣ Path lưu Vector DB
PERSIST_DIR = "/tmp/legal_chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

# 2️⃣ Load VectorStore
vectorstore = Chroma(persist_directory=PERSIST_DIR)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3️⃣ Prompt template
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
prompt_template = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# 4️⃣ Load DeepSeek client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 5️⃣ Hàm generate_answer
def generate_answer(question):
    # Lấy các document liên quan
    docs = retriever.retrieve(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Tạo prompt cuối cùng
    final_prompt = prompt_template.format(context=context, question=question)

    # Gọi DeepSeek API
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

# 6️⃣ Export
__all__ = ["retriever", "generate_answer"]
