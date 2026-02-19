from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv(r"C:\Users\2005k\OneDrive\Desktop\programs\LANGCHAIN\.env")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",  # safe public model
    temperature=0.2)

result = model.invoke('What is the capital of India')

print(result.content)
