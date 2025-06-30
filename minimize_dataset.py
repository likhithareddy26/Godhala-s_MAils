import pandas as pd

df = pd.read_csv('D:\\email_responder\\datasets\\enron_emails.csv')

# Reduce to 10,000 samples
sampled_df = df.sample(n=100, random_state=42)

# Save for training
sampled_df.to_csv('final_dataset.csv', index=False)
