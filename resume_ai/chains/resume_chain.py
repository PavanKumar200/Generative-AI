from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from prompts.templates import extract_prompt, evaluate_prompt

# Initialize the Groq LLM 
# Llama 3 70B is highly capable for evaluation and explanation logic
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    temperature=0
)

# Chain 1: Extract data from resume using LCEL
extraction_chain = extract_prompt | llm | StrOutputParser()

# Chain 2: Evaluate (Match + Score + Explain) using LCEL
evaluation_chain = evaluate_prompt | llm | StrOutputParser()