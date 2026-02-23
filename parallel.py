
# ✅ Stage 1 (Parallel):
# Model-1 → Generate a diet plan
# Model-2 → Generate workout questions
# ✅ Stage 2:
# Merge the diet + workout content into a single fitness guide document


from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model1 = ChatOpenAI(model="gpt-5-nano", temperature=0.8)
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# ---- PROMPTS ----

# Generate simple diet notes
prompt1 = PromptTemplate(
    template="Create short and simple healthy diet notes from the following context:\n{text}",
    input_variables=['text']
)

# Generate workout-related short Q&A
prompt2 = PromptTemplate(
    template="Generate 5 short workout-related question answers from the following context:\n{text}",
    input_variables=['text']
)

# Merge both outputs into a single fitness guide
prompt3 = PromptTemplate(
    template="Merge the diet notes and workout Q&A into a single, clean fitness guide.\n\nDiet Notes:\n{notes}\n\nWorkout Q&A:\n{quiz}",
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# ---- PARALLEL EXECUTION ----
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# ---- MERGE THE RESULTS ----
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

# ---- UNIQUE CONTEXT TEXT ----
text = """
A balanced fitness lifestyle includes both proper nutrition and structured exercise. 
Healthy eating focuses on whole foods such as fruits, vegetables, lean proteins, and whole grains. 
Hydration and calorie balance play an important role in maintaining energy levels.

A good workout routine includes cardio, strength training, and flexibility exercises. 
Beginners should start with lighter intensity workouts and progressively increase resistance. 
Rest and recovery are essential for muscle growth and injury prevention.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()
