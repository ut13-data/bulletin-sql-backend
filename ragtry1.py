# Step 2 + Step 3: Load and chunk the business definition markdown file

# Step 2: Read the file into a single string
with open(r'D:\DATAPAL\balaji-pharma-intelligence\Session Summaries.txt', 'r', encoding='utf-8') as file:
    fhand = file.read()

# Step 3: Chunk it using markdown headers as boundaries
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

headers_to_split_on = [("##", "section")]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(fhand)
# Inspect the result
#print("Total chunks:", chunks)


model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#all_vectors = model.embed_documents(chunks)   # chunks is already a list of strings

#print(len(all_vectors))
#print(len(all_vectors[0]))

vectorstore = FAISS.from_texts(chunks, model)
print(vectorstore.index.ntotal)

results = vectorstore.similarity_search("What is happening here?", 3)
retr = list()
for result in results :
     retr.append(result.page_content)

print(retr, len(retr))