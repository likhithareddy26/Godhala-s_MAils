import json

from preprocess_dataset import load_dataset

def prepare_data(output_path="final_data.jsonl", max_pairs=50):
    pairs = load_dataset()
    formatted_data = []

    for i, (email, reply) in enumerate(pairs[:max_pairs]):
        formatted_data.append({
            "prompt": f"### EMAIL:\n{email}\n\n### RESPONSE:\n",
            "completion": reply.strip()
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(formatted_data)} training samples to {output_path}")

if __name__ == "__main__":
    prepare_data()
