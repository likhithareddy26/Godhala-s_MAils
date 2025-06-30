import pandas as pd

def load_dataset(path="D:\\email_responder\\final_dataset.csv"):
    df = pd.read_csv(path)
    
    # Inspect and clean up
    df = df[['file', 'message']]  # adjust column names as needed
    df.dropna(inplace=True)

    # Optional: simulate email-response pairs (basic)
    email_pairs = []
    for i in range(0, len(df) - 1, 2):  # simple pairing
        email = df.iloc[i]['message']
        reply = df.iloc[i+1]['message']
        email_pairs.append((email, reply))

    return email_pairs

if __name__ == "__main__":
    pairs = load_dataset()
    print(f"Loaded {len(pairs)} email-response pairs.")
