from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
docs = [
    Document(page_content="Strength training helps build muscle and boost metabolism."),
    Document(page_content="Morning jogging improves cardiovascular endurance."),
    Document(page_content="A protein-rich diet supports muscle recovery."),
    Document(page_content="Meditation improves mental health and focus."),
    Document(page_content="HIIT workouts burn calories quickly."),
    Document(page_content="Salads provide fiber and essential micronutrients."),
]

embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}
)

query = "How can I stay fit and improve metabolism?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)