import os
import json
import requests
from dotenv import load_dotenv
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Using gemini-flash-latest as verified in previous steps
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-flash-latest")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")
SENT_JOBS_FILE = "sent_jobs.json"

# -------------------------------------
# Helper: Deduplication
# -------------------------------------

def load_sent_jobs():
    if os.path.exists(SENT_JOBS_FILE):
        try:
            with open(SENT_JOBS_FILE, "r") as f:
                return set(json.load(f))
        except (json.JSONDecodeError, ValueError):
            return set()
    return set()

def save_sent_jobs(job_ids):
    existing = load_sent_jobs()
    updated = existing.union(set(job_ids))
    with open(SENT_JOBS_FILE, "w") as f:
        json.dump(list(updated), f)

# -------------------------------------
# Helper: Resume Loading
# -------------------------------------

def load_resumes():
    files = ["backend_resume.tex", "fullstack_resume.tex", "frontend_resume.tex"]
    content = ""
    for file in files:
        if os.path.exists(file):
            content += f"\n\n--- Content from {file} ---\n"
            with open(file, "r", encoding="utf-8") as f:
                content += f.read()
    return content

# -------------------------------------
# 1. Fetch REAL jobs using SERPAPI
# -------------------------------------

def fetch_jobs(query):
    print(f"Searching: {query}")
    url = "https://serpapi.com/search.json"

    params = {
        "engine": "google_jobs",
        "q": query,
        "gl": "in",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    jobs = data.get("jobs_results", [])
    print(f"Found {len(jobs)} jobs for '{query}'")
    return jobs


# Combine multiple searches
def fetch_all_jobs():
    # Modified queries to be more specific about experience and location
    queries = [
        "Full Stack Developer 3 years experience Hyderabad",
        "React Developer 3 years experience Remote India",
        "Java Spring Boot Developer 3 years experience Hyderabad",
        "Python Backend Developer 3 years experience Remote India",
        "FastAPI Developer 3 years experience Remote India",
        "Django Developer 3 years experience Hyderabad",
        "Software Engineer 3 years experience Hyderabad"
    ]

    all_jobs = []
    sent_job_ids = load_sent_jobs()
    
    for q in queries:
        fetched = fetch_jobs(q)
        for job in fetched:
            # Use job_id if present, else construct a unique key from title+company
            job_id = job.get("job_id", f"{job.get('title')}-{job.get('company_name')}")
            
            if job_id not in sent_job_ids:
                all_jobs.append(job)
            else:
                # Optional: debug print for skipping
                pass

    print(f"Total new unique jobs: {len(all_jobs)}")
    return all_jobs


# -------------------------------------
# 2. Process using LangChain + Gemini
# -------------------------------------

def enrich_with_llm(jobs):
    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY to use Gemini.")

    # Load external prompt
    with open("job_prompt.txt", "r", encoding="utf-8") as f:
        base_prompt = f.read()
        
    # Load resumes
    resume_content = load_resumes()

    # Create the LLM instance
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.4
    )

    # Create Prompt Template
    # We include resumes block
    template = base_prompt + "\n\nMY RESUMES:\n{resumes}\n\nHere are the raw job listings:\n{jobs_json}"
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["jobs_json", "resumes"]
    )

    # Create Chain
    chain = prompt | llm | StrOutputParser()

    # Prepare input
    # Limit to e.g. 50 jobs to avoid token limits if too many returned
    jobs_sample = jobs[:50]
    jobs_json = json.dumps(jobs_sample, ensure_ascii=False)
    
    # Run Chain
    print("Invoking LangChain...")
    result = chain.invoke({"jobs_json": jobs_json, "resumes": resume_content})
    
    return result


# -------------------------------------
# 3. Send Email
# -------------------------------------

def send_email(html_body):
    print("Preparing email...")

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject="Daily Job Matches \u2013 3+ Years Exp (Filtered)",
        html_content=html_body
    )

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        sg.send(message)
        print("Email sent successfully!")
        return True
    except Exception as e:
        print("Email failed:", e)
        return False


# -------------------------------------
# MAIN WORKFLOW
# -------------------------------------

if __name__ == "__main__":
    print("Fetching jobs...")
    jobs = fetch_all_jobs()

    if len(jobs) == 0:
        print("No new jobs found (all duplicates or empty results).")
        exit()

    print(f"Processing {len(jobs)} jobs with LangChain (Gemini)...")
    enriched_html = enrich_with_llm(jobs)

    print("Sending email...")
    success = send_email(enriched_html)
    
    # If email sent successfully, mark jobs as sent
    if success:
        new_ids = []
        for job in jobs:
            job_id = job.get("job_id", f"{job.get('title')}-{job.get('company_name')}")
            new_ids.append(job_id)
        save_sent_jobs(new_ids)
        print(f"Saved {len(new_ids)} jobs to history.")
