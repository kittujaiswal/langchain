from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings ("ignore", category=UserWarning)
warnings.filterwarnings ("ignore", category=FutureWarning)
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens = None,
    timeout=None,
    max_retries=2,
)
result = model.invoke("write 5 lines poem on GenAI")
print("\n\n\n\n\n\n\n")
print(result.content)
print("\n\n\n\n\n\n\n")