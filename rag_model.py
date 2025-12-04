# rag_model.py
import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import OpenAI

PERSIST_DIR = "legal_chroma_db"
vectorstore = None
retriever = None

# --- NEW FUNCTION: Lazy Load ChromaDB ---
def get_retriever():
    """Tải và trả về retriever, chỉ khởi tạo vectorstore một lần."""
    global vectorstore
    global retriever
    
    # Nếu đã được khởi tạo, trả về ngay
    if retriever:
        return retriever
        
    print(f"BẮT ĐẦU: Khởi tạo ChromaDB từ thư mục: {PERSIST_DIR}")
    try:
        # Nếu thư mục không tồn tại, Chroma sẽ tự tạo (nhưng sẽ trống)
        # Sẽ không bị lỗi nếu DB trống, nhưng nên đảm bảo thư mục đã được commit.
        vectorstore = Chroma(persist_directory=PERSIST_DIR)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"THÀNH CÔNG: ChromaDB đã được load.")
        return retriever
    except Exception as e:
        print(f"LỖI KHỞI TẠO ChromaDB: {e}")
        # Đảm bảo ứng dụng KHÔNG bị crash khi load DB lỗi, chỉ báo lỗi khi dùng
        return None 
# ----------------------------------------


# ----------------------------------------------------
# Prompt template (Giữ nguyên)
# ----------------------------------------------------
PROMPT_TEMPLATE = """
... (Giữ nguyên)
"""
prompt_template = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ----------------------------------------------------
# Load DeepSeek client (Giữ nguyên)
# ----------------------------------------------------
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1" 
)

# ----------------------------------------------------
# Hàm generate_answer (Cập nhật)
# ----------------------------------------------------
def generate_answer(question):
    # Lấy retriever, sẽ khởi tạo DB nếu chưa có
    current_retriever = get_retriever()
    
    if not current_retriever:
        raise Exception("Retriever chưa được khởi tạo thành công. Vui lòng kiểm tra lại ChromaDB và thư mục commit.")
        
    # Lấy docs từ retriever
    docs = current_retriever.invoke(question)
    
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