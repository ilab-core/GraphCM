import pandas as pd
import json
import os
from collections import Counter

def analyze_query_distribution(data_file_path):
    """
    Veri setindeki sorguların (query_id) dağılımını analiz eder.
    """
    all_query_ids = []
    
    print(f"{data_file_path} dosyası okunuyor...")

    if not os.path.exists(data_file_path):
        print(f"HATA: Dosya bulunamadı: {data_file_path}")
        return

    with open(data_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            query_id = int(parts[1])
            all_query_ids.append(query_id)

    if not all_query_ids:
        print("Hiç sorgu bulunamadı.")
        return

    total_queries = len(all_query_ids)
    query_counts = Counter(all_query_ids)
    unique_query_count = len(query_counts)

    print("\n--- Sorgu Dağılım Analizi ---")
    print(f"Toplam Sorgu Sayısı (gösterim): {total_queries}")
    print(f"Benzersiz Sorgu Sayısı: {unique_query_count}")
    print("-" * 30)

    # Counter'ı sıralanabilir bir formata getirelim
    df = pd.DataFrame(query_counts.items(), columns=['query_id', 'count'])
    df['percentage'] = ((df['count'] / total_queries) * 100).round(2)
    df = df.sort_values(by='count', ascending=False).reset_index(drop=True)

    print("En Sık Tekrarlanan İlk 20 Sorgu:")
    print(df.head(20).to_string()) # to_string() ile tüm tabloyu düzgün göster
    print("-" * 30)

    # Top 10 sorgunun etkisini hesaplayalım
    top_10_percentage = df.head(10)['percentage'].sum()
    print(f"SONUÇ: En popüler 10 sorgu, tüm sorgu gösterimlerinin %{top_10_percentage:.2f}'sini oluşturuyor.")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    train_file = os.path.join(PROJECT_ROOT, 'data', 'emj', 'train_per_query_quid.txt')
    
    analyze_query_distribution(train_file)