import os
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from openai import OpenAI
from typing import List
import requests # Cần để gọi API
import json
from langchain_core.embeddings import Embeddings 

PERSIST_DIR = "legal_chroma_db"
vectorstore = None
retriever = None

# ----------------------------------------------------
# ĐỊNH NGHĨA CUSTOM EMBEDDING FUNCTION
# ----------------------------------------------------
class CustomLegalEmbedding(Embeddings):
    """Gửi yêu cầu embedding đến API Endpoint trên Hugging Face Spaces."""
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        self.headers = {"Content-Type": "application/json"}
        # Giả định mô hình truro7/vn-law-embedding có dimension 768
        self.expected_dim = 768 

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Thực hiện từng truy vấn nhỏ để tránh payload quá lớn
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        payload = {"text": text} 
        
        try:
            # Gửi request đến API Hugging Face
            response = requests.post(self.endpoint_url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status() 
            
            data = response.json()
            vector = data.get('embedding')
            
            if not vector:
                raise ValueError("API did not return 'embedding'.")

            # Kiểm tra kích thước vector để đảm bảo khớp DB
            if len(vector) != self.expected_dim:
                 raise ValueError(f"API returned dimension {len(vector)}, expected {self.expected_dim}.")

            return vector
        except Exception as e:
            # Nếu có lỗi, in ra và raise lại để hệ thống xử lý
            print(f"Lỗi gọi Custom Embedding API: {e}")
            raise

# ----------------------------------------------------
# KHỞI TẠO EMBEDDING API CUSTOM
# ----------------------------------------------------
# THAY THẾ URL NÀY BẰNG URL BẠN CÓ ĐƯỢC TỪ HUGGING FACE SPACES
HF_EMBEDDING_ENDPOINT = "https://[YOUR-USERNAME]-[YOUR-SPACE-NAME].hf.space/embed" 

try:
    embedding_function = CustomLegalEmbedding(endpoint_url=HF_EMBEDDING_ENDPOINT)
    print("Custom Embedding function (truro7/vn-law-embedding qua API) đã được tạo.")
except Exception as e:
    print(f"LỖI: Không thể khởi tạo Custom Embedding API: {e}")
    embedding_function = None


# --- NEW FUNCTION: Lazy Load ChromaDB ---
def get_retriever():
    """Tải và trả về retriever, chỉ khởi tạo vectorstore một lần."""
    global vectorstore
    global retriever
    
    if retriever:
        return retriever
    
    if embedding_function is None:
        print("LỖI: Embedding function API không tồn tại.")
        return None

    print(f"BẮT ĐẦU: Khởi tạo ChromaDB từ thư mục: {PERSIST_DIR} (Dùng Custom API)")
    try:
        # SỬ DỤNG CUSTOM EMBEDDING FUNCTION ĐÃ KHÔNG CẦN TẢI MODEL NẶNG
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_function 
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print(f"THÀNH CÔNG: ChromaDB đã được load.")
        return retriever
    except Exception as e:
        print(f"LỖI KHỞI TẠO ChromaDB: {e}")
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