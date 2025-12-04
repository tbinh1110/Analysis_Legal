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

def generate_answer(question):
    # 1. Truy xuất ngữ cảnh từ VectorStore
    relevant_docs = retriever.get_relevant_documents(question)  # trả về list[Document]
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 2. Tạo prompt cuối cùng
    final_prompt = prompt.format(context=context, question=question)

    # 3. Gọi DeepSeek API
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý pháp lý AI."},
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.2,
        max_tokens=512
    )

    # 4. Trả kết quả
    return response.choices[0].message["content"]

