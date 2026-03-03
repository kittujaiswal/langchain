from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
chatmodel = ChatOpenAI(model="gpt-5-nano",temperature = 0.8)
print("Let's start chat write quit to terminate conversation \n")
while True:
    message = input("You : ")
    history=[]
    if message.lower()=="quit":
        break
    response= chatmodel.invoke(history)
    history.append(response.content)
    print("Bot : ",response.content)
print(history)