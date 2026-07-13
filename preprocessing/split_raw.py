import pandas as pd
from sklearn.model_selection import train_test_split
import os

raw_dir = 'student_raw'
df = pd.read_csv(os.path.join(raw_dir, 'train.csv'))

# Create target before splitting, or just split raw data directly
# It's better to split raw data directly so it truly simulates raw train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_csv(os.path.join(raw_dir, 'train.csv'), index=False)
test_df.to_csv(os.path.join(raw_dir, 'test.csv'), index=False)
print("Raw data has been split into train and test successfully.")
