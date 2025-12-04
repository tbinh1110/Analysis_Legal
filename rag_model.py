import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from openai import OpenAI


PERSIST_DIR = "legal_chroma_db"

# ----------------------------------------------------
# 2Load VectorStore
# ----------------------------------------------------
# Lưu ý: Bạn cần đảm bảo đã tạo sẵn vector DB này và commit lên GitHub.
try:
    # Nếu thư mục không tồn tại, Chroma sẽ tự tạo (nhưng sẽ trống)
    vectorstore = Chroma(persist_directory=PERSIST_DIR)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print(f"ChromaDB đã được load từ thư mục: {PERSIST_DIR}")
except Exception as e:
    # Trường hợp thư viện bị lỗi hoặc không tìm thấy file
    print(f"Lỗi khi load ChromaDB: {e}")
    # Đặt retriever thành None để tránh lỗi crash ngay lập tức
    retriever = None

# ----------------------------------------------------
# Prompt template
# ----------------------------------------------------
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

# ----------------------------------------------------
# Load DeepSeek client
# ----------------------------------------------------
# Lấy API Key từ biến môi trường (Render Env Vars)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"  # Cổng chuẩn là /v1
)

# ----------------------------------------------------
# Hàm generate_answer 
# ----------------------------------------------------
def generate_answer(question):
    if not retriever:
        raise Exception("Retriever chưa được khởi tạo thành công. Vui lòng kiểm tra lại ChromaDB.")
        
    #
    docs = retriever.invoke(question)
    
    # Tạo context từ các document đã lấy được
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

    return response.choices[0].message.content


__all__ = ["generate_answer"]