import os
from dotenv import load_dotenv

# 1. Load environment variables FIRST
load_dotenv()

# 2. THEN import your chains
from chains.resume_chain import extraction_chain, evaluation_chain

def read_file(filepath):
    """Helper function to read text files."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return ""

def screen_candidate(resume_text, jd_text):
    """Executes the pipeline: Extract -> Match -> Score -> Explain"""
    
    if not resume_text:
        raise ValueError("Resume text is empty. Cannot process.")

    # Step 1: Extract Skills, Experience, Tools
    print("Extracting profile...")
    extracted_profile = extraction_chain.invoke({"resume_text": resume_text})
    
    # Step 2, 3 & 4: Match, Score, and Explain
    print("Evaluating candidate...")
    final_evaluation = evaluation_chain.invoke({
        "job_description": jd_text,
        "candidate_profile": extracted_profile
    })
    
    return final_evaluation

if __name__ == "__main__":
    print("--- Starting AI Resume Screening System (Powered by Groq) ---")
    
    # Load the Job Description
    job_description = read_file("data/job_description.txt")
    
    # Load Resumes
    resumes = {
        "Strong": read_file("data/resume_strong.txt"),
        "Average": read_file("data/resume_average.txt"),
        "Weak": read_file("data/resume_weak.txt")
    }
    
    # Process each candidate to generate LangSmith traces
    for candidate_type, resume_text in resumes.items():
        print(f"\nEvaluating {candidate_type} Candidate...")
        if resume_text:
            result = screen_candidate(resume_text, job_description)
            print("\nResult:")
            print(result)
        else:
            print(f"File for {candidate_type} candidate not found. Please add it to data/ folder.")
        print("-" * 50)

    # Triggering an intentional error for LangSmith debugging requirement
    print("\n--- Triggering Intentional Error for LangSmith ---")
    try:
        screen_candidate("", job_description) 
    except Exception as e:
        print(f"Pipeline failed as expected (Check LangSmith for the error trace): {e}")