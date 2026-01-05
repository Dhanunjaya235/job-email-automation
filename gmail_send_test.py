import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# MUST match how token.json was generated
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

FROM_EMAIL = "dhanunjaya512000@gmail.com"
TO_EMAIL = "dhanunjaya.andavarapu@outlook.com"  # send to self for test

def send_test_email():
    # Load token
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # Build Gmail service
    service = build("gmail", "v1", credentials=creds)

    # Create message
    msg = MIMEText("✅ Gmail API test email – if you got this, auth is working.")
    msg["To"] = TO_EMAIL
    msg["From"] = FROM_EMAIL
    msg["Subject"] = "Gmail API Test"

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    # Send
    result = service.users().messages().send(
        userId="me",
        body={"raw": raw}
    ).execute()

    print("SUCCESS. Message ID:", result["id"])


if __name__ == "__main__":
    send_test_email()
