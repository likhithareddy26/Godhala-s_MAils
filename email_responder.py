from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

model_path = "./gpt2-finetuned-email"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_response(email):
    prompt = f"### EMAIL:\n{email}\n\n### RESPONSE:\n"
    result = generator(prompt, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
    return result[0]["generated_text"].split("### RESPONSE:\n")[-1].strip()

if __name__ == "__main__":
    email = input("Paste an email:\n")
    print("\nGenerated Response:\n", generate_response(email))
