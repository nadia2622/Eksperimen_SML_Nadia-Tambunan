import pandas as pd
import os
import argparse
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = 'Label Kelulusan (0/1)'

def load_data(raw_data_dir):
    """Membaca data train dan test dari folder raw."""
    train_path = os.path.join(raw_data_dir, 'train.csv')
    test_path = os.path.join(raw_data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess(train_df, test_df):
    """Melakukan tahapan preprocessing data untuk dataset Student Performance."""
    
    # 1. Binarize Target: Lulus (1) jika G3 >= 10, else Tidak Lulus (0)
    if 'G3' in train_df.columns:
        train_df[TARGET_COL] = train_df['G3'].apply(lambda x: 1 if x >= 10 else 0)
        test_df[TARGET_COL] = test_df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    
    # 2. Drop kolom nilai agar tidak terjadi data leakage (kebocoran data)
    cols_to_drop = ['G1', 'G2', 'G3']
    train_df = train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns])
    
    # 3. Label Encoding untuk kolom kategorikal
    cat_cols = train_df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        # Handle potensi unseen labels di test data
        # Mengubah unseen labels jadi nilai modus di train data
        test_df[col] = test_df[col].apply(lambda x: x if x in le.classes_ else train_df[col].mode()[0])
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        test_df[col] = test_df[col].map(le_dict).fillna(train_df[col].mode()[0]).astype(int)
        
    # 4. Standard Scaling untuk kolom numerik (kecuali target)
    num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns
    num_cols = [c for c in num_cols if c != TARGET_COL]
    
    scaler = StandardScaler()
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
            
    return train_df, test_df

def save_data(train_df, test_df, output_dir):
    """Menyimpan data hasil preprocessing."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print(f"Data preprocessing berhasil disimpan di direktori: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Otomatisasi Preprocessing SML")
    parser.add_argument('--input_dir', type=str, default='student_raw', help='Folder data raw')
    parser.add_argument('--output_dir', type=str, default='student_preprocessing', help='Folder data preprocessing')
    args = parser.parse_args()
    
    print(f"Memulai preprocessing dari {args.input_dir}...")
    
    if not os.path.exists(args.input_dir):
        print(f"Folder {args.input_dir} tidak ditemukan. Harap pastikan dataset diletakkan di sana.")
        return
        
    train_df, test_df = load_data(args.input_dir)
    train_clean, test_clean = preprocess(train_df, test_df)
    save_data(train_clean, test_clean, args.output_dir)
    print("Selesai!")

if __name__ == '__main__':
    main()
