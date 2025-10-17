# ilan_sayisi_analizi.py (Yüzde Hesaplama Eklendi)

import argparse
import os
import re
from tqdm import tqdm
from collections import Counter

def analyze_listing_distribution(filepath, set_name):
    """
    Belirtilen log dosyasını analiz eder ve sorgu başına düşen ilan sayısının
    dağılımını çoktan aza doğru sıralayarak basar.
    """
    print(f"\n--- {set_name} Analizi Başlatılıyor: {filepath} ---")

    if not os.path.exists(filepath):
        print(f"HATA: Belirtilen yolda dosya bulunamadı: {filepath}")
        return

    listing_counts = Counter()
    total_q_count = 0

    try:
        print("-> Toplam satır sayısı hesaplanıyor...")
        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"   {total_lines:,} satır bulundu.")

        print("-> Dosya içeriği analiz ediliyor...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc=f"   {set_name} İşleniyor"):
                parts = re.split(r'\s+', line.strip())

                if len(parts) > 2 and parts[2] == 'Q':
                    total_q_count += 1
                    
                    # İlanlar 5 metaveri sütunundan sonra başladığı için 5 çıkarıyoruz.
                    num_listings = len(parts) - 5
                    
                    if num_listings > 0:
                        listing_counts[num_listings] += 1

    except Exception as e:
        print(f"HATA: Dosya okunurken bir sorun oluştu: {e}")
        return

    print(f"\n--- {set_name} Analiz Sonuçları ---")
    
    if total_q_count > 0:
        print(f"Toplam Sorgu (Q) Sayısı: {total_q_count:,}")
        print("-" * 40)

        sorted_counts = sorted(listing_counts.items(), key=lambda item: item[0], reverse=True)

        for num_listings, query_count in sorted_counts:
            print(f"{num_listings} ilan olan {query_count:,} query var")
            
        print("-" * 40)

        # 30 ilana sahip sorguların yüzdesini hesapla ve yazdır
        queries_with_30_listings = listing_counts.get(30, 0)
        if queries_with_30_listings > 0:
            percentage_30_listings = (queries_with_30_listings / total_q_count) * 100
            print(f"Önemli Not: 30 ilana sahip sorgular, "
                  f"tüm sorguların %{percentage_30_listings:.2f}'ini oluşturmaktadır.")
            print("-" * 40)

    else:
        print("Dosyada hiç geçerli sorgu (Q) satırı bulunamadı.")

def main():
    parser = argparse.ArgumentParser(
        description="Train ve Test log dosyalarındaki sorgu başına düşen ilan sayısının dağılımını analiz eder."
    )
    parser.add_argument('--train_file', required=True, help="Analiz edilecek training log dosyasının yolu.")
    parser.add_argument('--test_file', required=True, help="Analiz edilecek test log dosyasının yolu.")
    args = parser.parse_args()

    analyze_listing_distribution(args.train_file, "Train Set")
    analyze_listing_distribution(args.test_file, "Test Set")

if __name__ == "__main__":
    main()