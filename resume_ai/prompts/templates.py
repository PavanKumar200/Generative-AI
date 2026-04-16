from langchain_core.prompts import PromptTemplate

# Step 1: Skill Extraction Prompt
extract_template = """
You are an expert technical recruiter. Extract the following from the provided resume:
1. Skills
2. Experience (in years)
3. Tools

CRITICAL RULE: Do NOT assume skills not present in the resume. Only extract explicit matches.

Resume:
{resume_text}

Output as a structured JSON with keys: "skills", "experience", "tools".
"""
extract_prompt = PromptTemplate.from_template(extract_template)

# Step 2 & 3: Matching, Scoring, and Explaining Prompt
evaluate_template = """
You are an AI Resume Screener. Evaluate the extracted candidate profile against the Job Description.

Job Description:
{job_description}

Candidate Profile (Extracted):
{candidate_profile}

Task:
1. Match the candidate's skills and experience against the job requirements.
2. Assign a fit score from 0 to 100.
3. Provide a detailed explanation of why this score was assigned.

Output Format:
Score: [0-100]
Explanation: [Your detailed reasoning]
"""
evaluate_prompt = PromptTemplate.from_template(evaluate_template)