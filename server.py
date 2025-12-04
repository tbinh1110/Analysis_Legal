# server.py
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
        # Retrieve legal context
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([d.page_content for d in docs])

        # Generate answer via DeepSeek
        answer = generate_answer(context, query)

        return {
            "status": "success",
            "result": answer
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
