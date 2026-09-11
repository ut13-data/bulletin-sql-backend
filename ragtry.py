# Step 2 + Step 3: Load and chunk the business definition markdown file
from dotenv import load_dotenv
load_dotenv()

import os
groq_key = os.getenv("GROQ_API_KEY")
from groq import Groq

client = Groq(api_key=groq_key)

# Step 2: Read the file into a single string
with open(r'D:\DATAPAL\balaji-pharma-intelligence\Balaji_Pharma_Database_Architecture.md', 'r', encoding='utf-8') as file:
    fhand = file.read()

# Step 3: Chunk it using markdown headers as boundaries
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

headers_to_split_on = [("##", "section")]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
chunks = splitter.split_text(fhand)

# Inspect the result
print("Total chunks:", len(chunks))
print()

for i, chunk in enumerate(chunks):
    section = chunk.metadata.get("section", "(intro/no section)")
    print(f"Chunk {i}: [{section}] -- {len(chunk.page_content)} chars")



content = list()
for chunk in chunks :
    content.append(chunk.page_content)
   #clear print (content)




model = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
#all_vectors = model.embed_documents(content)
#print(len(all_vectors))
#print(len(all_vectors[0]))
vectorstore = FAISS.from_texts(content, model)
#print(vectorstore.index.ntotal)

results = vectorstore.similarity_search("What is the flow of this company?", 3)
retr = list()
for result in results :
    retr.append(result.page_content)

#print(retr, len(retr))
question = "What is the DistributionCluster?"
prompt = f"Context: \n\n{retr[0]}\n{retr[1]}\n{retr[2]}\n\nUsing only the context above, answer the question. If the answer isn't in the context, say you don't know.\n\nQuestion: {question}"
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print(response.choices[0].message.content)