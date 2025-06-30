# email_handler.py

import os
import pickle
from base64 import urlsafe_b64encode
from email.mime.text import MIMEText
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Define the SCOPES
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    """Authenticate and create a Gmail API service."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f"❌ An error occurred while building Gmail service: {error}")
        return None

def create_message(sender, recipient, subject, body):
    """Create a MIME email message."""
    message = MIMEText(body)
    message['to'] = recipient
    message['from'] = sender
    message['subject'] = subject
    raw_message = urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    return {'raw': raw_message}

def send_message(service, user_id, message):
    """Send an email message via the Gmail API."""
    try:
        sent_message = service.users().messages().send(userId=user_id, body=message).execute()
        print(f"✅ Message sent successfully! Message ID: {sent_message['id']}")
    except HttpError as error:
        print(f"❌ An error occurred while sending the email: {error}")

def send_email(service, recipient, subject, body):
    """Create and send an email."""
    try:
        sender = "me"  # 'me' indicates the authenticated user
        message = create_message(sender, recipient, subject, body)
        send_message(service, sender, message)
        print(f"✅ Email sent to {recipient}")
    except Exception as e:
        print(f"❌ Failed to send email to {recipient}: {e}")

def fetch_unread_emails(service):
    """Fetch unread emails from Gmail."""
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
        unread_emails = []

        if not messages:
            print("No unread messages.")
        else:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id']).execute()
                headers = msg['payload'].get('headers', [])
                from_email = None
                subject = "(No Subject)"
                for header in headers:
                    if header['name'] == 'From':
                        from_email = header['value']
                    if header['name'] == 'Subject':
                        subject = header['value']
                
                if from_email:
                    unread_emails.append({
                        'id': message['id'],
                        'from': from_email,
                        'subject': subject,
                        'snippet': msg.get('snippet', '')
                    })

        return unread_emails
    except HttpError as error:
        print(f"❌ An error occurred while fetching emails: {error}")
        return None
def mark_as_read(service, message_id):
    """Mark an email as read using Gmail API."""
    try:
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print(f"📬 Marked message {message_id} as read.")
    except HttpError as error:
        print(f"❌ Error marking email as read: {error}")
