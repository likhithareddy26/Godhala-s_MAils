from transformers import pipeline
'''
# Load a pre-trained transformer for text generation
generator = pipeline("text-generation", model="gpt2")

def generate_response(email_text):
    prompt = f"Reply to the following email professionally:\n\n{email_text}\n\nResponse:"
    response = generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    return response[0]["generated_text"].split("Response:")[-1].strip()
'''

# Load the fine-tuned model
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
    # Improved response extraction
    generated_text = response[0]["generated_text"]
    # Find the last occurrence of 'Response:' and extract everything after it
    if "Response:" in generated_text:
        reply = generated_text.split("Response:")[-1].strip()
        # Remove any repeated input text from the reply
        if email_text in reply:
            reply = reply.replace(email_text, "").strip()
        return reply
    return "I apologize, but I couldn't generate a proper response. Please try again."