import os
import csv
import openai
from datetime import datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
import base64

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL")
TO_EMAIL = os.getenv("TO_EMAIL")

openai.api_key = OPENAI_API_KEY

# --- Load prompt ---
with open("job_prompt.txt", "r", encoding="utf-8") as f:
    JOB_PROMPT = f.read()


def fetch_jobs():
    """Call OpenAI to get job recommendations."""
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You find jobs and format them cleanly."},
            {"role": "user", "content": JOB_PROMPT}
        ]
    )
    return response.choices[0].message["content"]


def create_csv(text_output):
    """Create CSV from job listings in plain text."""
    rows = []
    for block in text_output.split("\n\n"):
        if "—" in block and "http" in block:
            parts = block.split("\n")
            title_line = parts[0]
            skills = parts[1] if len(parts) > 1 else ""
            reason = parts[2] if len(parts) > 2 else ""
            link = parts[-1] if "http" in parts[-1] else ""

            rows.append([title_line, skills, reason, link])

    filename = "jobs.csv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Job Title", "Skills", "Match Reason", "Apply Link"])
        writer.writerows(rows)

    return filename


def send_email(body_text, csv_file):
    """Send the email with CSV attachment."""
    subject = f"Daily Job Matches — {datetime.now().strftime('%Y-%m-%d')} (09:00 IST)"

    message = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=subject,
        html_content=f"<pre>{body_text}</pre>",
        plain_text_content=body_text
    )

    # Attach CSV
    with open(csv_file, 'rb') as f:
        data = f.read()

    encoded = base64.b64encode(data).decode()

    attachment = Attachment(
        file_content=FileContent(encoded),
        file_type=FileType('text/csv'),
        file_name=FileName('jobs.csv'),
        disposition=Disposition('attachment')
    )

    message.attachment = attachment

    sg = SendGridAPIClient(SENDGRID_API_KEY)
    sg.send(message)


if __name__ == "__main__":
    try:
        print("Fetching jobs...")
        jobs = fetch_jobs()

        print("Creating CSV...")
        csv_file = create_csv(jobs)

        print("Sending Email...")
        send_email(jobs, csv_file)

        print("Job email sent successfully!")
    except Exception as e:
        print("Error:", e)
