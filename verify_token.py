from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]

creds = Credentials.from_authorized_user_file("token.json", SCOPES)

print("Token valid:", creds.valid)
print("Token scopes:", creds.scopes)
print("Token expired:", creds.expired)
