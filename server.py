import os
from fastapi import FastAPI
from pydantic import BaseModel
from rag_model import generate_answer 

app = FastAPI()

class ContractInput(BaseModel):
    contract_text: str

@app.post("/analyze")
def analyze_contract(data: ContractInput):
    query = data.contract_text

    # Kiểm tra API Key
    if not os.getenv("DEEPSEEK_API_KEY"):
         return {
            "status": "error", 
            "message": "Lỗi cấu hình: Biến môi trường DEEPSEEK_API_KEY không được tìm thấy."
        }

    try:
        print("Đang sinh phân tích hợp đồng...")

        # Gọi trực tiếp hàm generate_answer
        answer = generate_answer(query)

        return {
            "status": "success",
            "message": "Phân tích hoàn tất!",
            "result": answer
        }

    except Exception as e:
        # In lỗi chi tiết ra console Render để debug
        print(f"🔥 LỖI CHÍNH XÁC: {str(e)}")
        
        return {
            "status": "error",
            "message": f"Có lỗi xảy ra: {str(e)}"
        }