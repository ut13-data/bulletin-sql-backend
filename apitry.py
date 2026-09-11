# ============================================================
# imports
# ============================================================
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq
import os

# ============================================================
# groq client setup
# ============================================================
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_key)

# ============================================================
# reading the files
# ============================================================
with open(r'D:\DATAPAL\bUlleTin\bulletin-sql-backend\docs\Balaji_Pharma_Database_Architecture.md', 'r', encoding='utf-8') as f1, \
     open(r'D:\DATAPAL\bUlleTin\bulletin-sql-backend\docs\Balaji_Pharma_Business_Definition.md', 'r', encoding='utf-8') as f2:
    t1 = f1.read()
    t2 = f2.read()

# ============================================================
# splitting the files (different header depth per doc)
# ============================================================
headers_architecture = [("##", "section"), ("###", "subsection")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_architecture)
chunk1 = splitter.split_text(t1)

headers_business = [("##", "section")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_business)
chunk2 = splitter.split_text(t2)

content = list()
for chunk in chunk1:
    content.append(chunk.page_content)

for chunk in chunk2:
    content.append(chunk.page_content)

# ============================================================
# embeddings model + vectorstore
# ============================================================
model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.from_texts(content, model)

# ============================================================
# retrieval (test question, hardcoded for now)
# ============================================================
question = "What is the flow of this company?"

results = vectorstore.similarity_search(question, k=3)

retr = list()
for result in results:
    retr.append(result.page_content)

# ============================================================
# augmentation + generation
# ============================================================
prompt = f"Context: \n\n{retr[0]}\n{retr[1]}\n{retr[2]}\n\nUsing only the context above, answer the question. If the answer isn't in the context, say you don't know.\n\nQuestion: {question}"

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)