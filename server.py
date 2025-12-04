import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_model import generate_answer  # chỉ cần import generate_answer

app = FastAPI()

class ContractInput(BaseModel):
    contract_text: str

@app.post("/analyze")
def analyze_contract(data: ContractInput):
    query = data.contract_text

    try:
        print("✅ Đang sinh phân tích hợp đồng...")

        # Gọi trực tiếp hàm generate_answer (hàm này tự truy xuất từ retriever)
        answer = generate_answer(query)

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
