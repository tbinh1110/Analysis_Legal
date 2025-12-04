import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_model import retriever, generate_answer

app = FastAPI()

class ContractInput(BaseModel):
    contract_text: str

@app.post("/analyze")
def analyze_contract(data: ContractInput):
    query = data.contract_text

    try:
        # --- Bước 1: Lấy các tài liệu liên quan ---
        # Sử dụng get_relevant_documents nếu retriever hỗ trợ
        # Nếu retriever version mới, thử get_relevant_texts
        try:
            docs = retriever.get_relevant_documents(query)
            context = "\n\n".join([d.page_content for d in docs])
        except AttributeError:
            # fallback cho phiên bản mới của langchain
            texts = retriever.get_relevant_texts(query)
            context = "\n\n".join(texts)

        # --- Bước 2: Thông báo đang xử lý ---
        print("✅ Đang sinh phân tích hợp đồng...")

        # --- Bước 3: Gọi DeepSeek API ---
        answer = generate_answer(context, query)

        # --- Bước 4: Trả kết quả ---
        return {
            "status": "success",
            "message": "✅ Phân tích hoàn tất!",
            "result": answer
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Có lỗi xảy ra: {str(e)}"
        }
