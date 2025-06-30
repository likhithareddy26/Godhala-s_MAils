from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import base64

# Scope for read-only Gmail access

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.send']


def authenticate_gmail():
    creds = None
    try:
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    except:
        pass
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def extract_body(payload):
    """Extracts the body from the email payload."""
    if 'data' in payload.get('body', {}):
        body_data = payload['body']['data']
        body = base64.urlsafe_b64decode(body_data).decode('utf-8')
        return body
    elif 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'text/plain':
                body_data = part['body'].get('data')
                if body_data:
                    body = base64.urlsafe_b64decode(body_data).decode('utf-8')
                    return body
    return 'No body content.'

def get_latest_emails(service, num_emails=5):
    results = service.users().messages().list(userId='me', maxResults=num_emails).execute()
    messages = results.get('messages', [])

    email_list = []

    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            payload = msg['payload']
            headers = payload['headers']

            subject = ''
            for header in headers:
                if header['name'] == 'Subject':
                    subject = header['value']
                    break  # No need to continue once found

            body = extract_body(payload)

            email_list.append({'subject': subject, 'body': body})

    return email_list

if __name__ == '__main__':
    creds = authenticate_gmail()
    service = build('gmail', 'v1', credentials=creds)

    emails = get_latest_emails(service, num_emails=5)
    for idx, email_data in enumerate(emails):
        print(f"\n--- Email {idx + 1} ---")
        print(f"Subject: {email_data['subject']}")
        print(f"Body: {email_data['body'][:500]}")
