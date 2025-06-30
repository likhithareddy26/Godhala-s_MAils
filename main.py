# main.py

from email_handler import fetch_unread_emails, send_email, authenticate_gmail, mark_as_read
from responder import generate_response

def is_auto_reply(email_address):
    return 'no-reply' in email_address.lower() or 'noreply' in email_address.lower()

def run_responder():
    try:
        service = authenticate_gmail()
        if not service:
            print("❌ Failed to authenticate Gmail service.")
            return

        emails = fetch_unread_emails(service)
        if not emails:
            print("📭 No unread emails found.")
            return

        # Process only the most recent email
        latest_email = emails[0]
        sender = latest_email.get('from')
        subject = latest_email.get('subject', 'No Subject')
        snippet = latest_email.get('snippet', '')
        msg_id = latest_email.get('id')

        if is_auto_reply(sender):
            print(f"⚠️ Skipping auto-generated email from {sender}")
            return

        print(f"🧠 Generating reply for email from {sender}...")
        reply = generate_response(snippet)

        if reply:
            reply_subject = f"Re: {subject}"
            send_email(service, sender, reply_subject, reply)
            mark_as_read(service, msg_id)
            print(f"✅ Sent reply to {sender}\n")
        else:
            print(f"⚠️ No reply generated for email from {sender}\n")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    run_responder()
