from transformers import pipeline, AutoModelForCausalLM, GPT2Tokenizer

# Load the fine-tuned model
model_path = "D:\\email_responder\\gpt2-finetuned-email"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Initialize the text generation pipeline
generator = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    device=-1  # Ensure using CPU
)

def generate_reply(email_text, max_new_tokens=50):
    prompt = f"Reply to this email:\n\n{email_text}\n\nResponse:"
    response = generator(
        prompt,
        max_new_tokens=max_new_tokens,  # Limits the number of new tokens generated
        num_return_sequences=1,
        truncation=True,
        temperature=0.7,    # More natural creativity
        top_p=0.9           # Nucleus sampling for better coherence
    )
    generated_text = response[0]['generated_text']
    reply = generated_text.replace(prompt, '').strip()
    return reply

if __name__ == "__main__":
    test_email = "Hello, could you send me the documents by tomorrow?"
    reply = generate_reply(test_email)
    print("\nGenerated reply:\n", reply)
