# Logic:
# If the user text is about technology, generate:
# Tech notes using Model-1
# Tech quiz using Model-2
# If the user text is about health, generate:
# Health notes using Model-1
# Health quiz using Model-2
# Then merge both outputs into a single documen

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel
load_dotenv()
model1 = ChatOpenAI(model="gpt-5-nano", temperature=0.8)
model2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   
)
parser = StrOutputParser()
# ---- PROMPTS FOR DIFFERENT DOMAINS ----
tech_notes = PromptTemplate(
    template="Write simple and short TECH notes from the text:\n{text}",
    input_variables=["text"]
)

tech_quiz = PromptTemplate(
    template="Generate 5 TECH-related question answers from the text:\n{text}",
    input_variables=["text"]
)

health_notes = PromptTemplate(
    template="Write simple and short HEALTH notes from the text:\n{text}",
    input_variables=["text"]
)

health_quiz = PromptTemplate(
    template="Generate 5 HEALTH-related question answers from the text:\n{text}",
    input_variables=["text"]
)

merge_prompt = PromptTemplate(
    template="Merge the notes and quiz into a single clean document.\n\nNotes:\n{notes}\n\nQuiz:\n{quiz}",
    input_variables=["notes", "quiz"]
)


# ---- CONDITIONAL LOGIC (BRANCH) ----
def classify_topic(input_dict):
    text = input_dict["text"].lower()
    if any(word in text for word in ["ai", "technology", "computer", "software"]):
        return "tech"
    return "health"


tech_chain = RunnableParallel({
    "notes": tech_notes | model1 | parser,
    "quiz": tech_quiz | model2 | parser
})

health_chain = RunnableParallel({
    "notes": health_notes | model1 | parser,
    "quiz": health_quiz | model2 | parser
})

conditional_chain = RunnableBranch(
    (lambda x: classify_topic(x) == "tech", tech_chain),
    health_chain   # default condition
)

# ---- FINAL MERGE ----
final_chain = conditional_chain | (merge_prompt | model2 | parser)


# ---- TESTING INPUT ----
text = """
Artificial Intelligence is transforming industries. Machines can now learn patterns, 
analyze large data, and make decisions. Modern AI models use neural networks, transformers,
and reinforcement learning to optimize performance.
"""

result = final_chain.invoke({"text": text})
print(result)


final_chain.get_graph().print_ascii()