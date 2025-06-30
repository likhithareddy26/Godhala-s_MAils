import json
import re

def extract_body(text):
    """Extracts the body from raw email by removing headers."""
    split = re.split(r"\n\s*\n", text, maxsplit=1)
    return split[1].strip() if len(split) > 1 else text.strip()

def clean_data(input_path="training_data.jsonl", output_path="cleaned_data.jsonl", max_pairs=1000):
    with open(input_path, "r", encoding="utf-8") as infile:
        raw_lines = [json.loads(line) for line in infile]

    cleaned_lines = []
    for line in raw_lines[:max_pairs]:
        email = extract_body(line["prompt"])
        response = extract_body(line["completion"])

        if len(email) < 20 or len(response) < 20:
            continue  # skip very short samples

        cleaned_lines.append({
            "prompt": f"### EMAIL:\n{email}\n\n### RESPONSE:\n",
            "completion": response
        })

    with open(output_path, "w", encoding="utf-8") as outfile:
        for item in cleaned_lines:
            outfile.write(json.dumps(item) + "\n")

    print(f"✅ Cleaned and saved {len(cleaned_lines)} pairs to {output_path}")

if __name__ == "__main__":
    clean_data()
