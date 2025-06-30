from transformers import pipeline
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from email.mime.text import MIMEText
from base64 import urlsafe_b64encode
import os

# Define the SCOPES for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.send', 'https://www.googleapis.com/auth/gmail.readonly']

# Load the fine-tuned GPT-2 model for email response generation
model_path = "gpt2-finetuned-email"
generator = pipeline("text-generation", model=model_path)

def generate_response(email_text):
    prompt = f"Reply to the following email professionally:\n\n{email_text}\n\nResponse:"
    response = generator(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1
    )
    generated_text = response[0]["generated_text"]
    if "Response:" in generated_text:
        reply = generated_text.split("Response:")[-1].strip()
        if email_text in reply:
            reply = reply.replace(email_text, "").strip()
        return reply
    return "I apologize, but I couldn't generate a proper response. Please try again."

def authenticate_gmail():
    """Authenticate and create a Gmail API service without persistent token storage."""
    creds = None
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except HttpError as error:
        print(f"An error occurred while building Gmail service: {error}")
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
        print(f"Message sent successfully! Message ID: {sent_message['id']}")
    except HttpError as error:
        print(f"An error occurred while sending the email: {error}")

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
                snippet = msg.get('snippet', '')
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
                        'snippet': snippet
                    })
        return unread_emails
    except HttpError as error:
        print(f"An error occurred while fetching emails: {error}")
        return None

def mark_as_read(service, message_id):
    """Mark an email as read using Gmail API."""
    try:
        service.users().messages().modify(
            userId='me',
            id=message_id,
            body={'removeLabelIds': ['UNREAD']}
        ).execute()
        print(f"Marked message {message_id} as read.")
    except HttpError as error:
        print(f"Error marking email as read: {error}")

def main():
    service = authenticate_gmail()
    if not service:
        print("Failed to authenticate Gmail service.")
        return

    unread_emails = fetch_unread_emails(service)
    if not unread_emails:
        print("No unread emails to respond to.")
        return

    for email in unread_emails:
        print(f"Processing email from: {email['from']} with subject: {email['subject']}")
        response_text = generate_response(email['snippet'])
        print(f"Generated response:\n{response_text}\n")
        message = create_message("me", email['from'], f"Re: {email['subject']}", response_text)
        send_message(service, "me", message)
        mark_as_read(service, email['id'])

if __name__ == "__main__":
    main()
