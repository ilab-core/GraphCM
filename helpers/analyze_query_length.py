# analyze_query_length.py (Boşluk/Tab Hatalarına Karşı En Dayanıklı Versiyon)

import argparse
import os
import re # Regular Expression modülünü ekliyoruz
from tqdm import tqdm

def analyze_log_file(filepath):
    """
    Belirtilen log dosyasını analiz eder ve sorgu uzunlukları hakkında istatistikler basar.
    """
    print(f"Dosya analizi başlatılıyor: {filepath}")

    if not os.path.exists(filepath):
        print(f"HATA: Belirtilen yolda dosya bulunamadı: {filepath}")
        return

    total_q_count = 0
    q_gt_10_count = 0  # Kırpılacaklar: İlan sayısı > 10
    q_lt_10_count = 0  # Doldurulacaklar: İlan sayısı < 10
    q_eq_10_count = 0  # Tam Uyanlar: İlan sayısı == 10

    try:
        print("-> Toplam satır sayısı hesaplanıyor...")
        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        print(f"   {total_lines:,} satır bulundu.")
    except Exception as e:
        print(f"HATA: Dosya okunurken bir sorun oluştu: {e}")
        return

    print("-> Dosya içeriği analiz ediliyor...")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="   İşleniyor"):
            # --- NİHAİ DÜZELTME BURADA ---
            # Satırı, tab/boşluk fark etmeksizin tüm boşluk karakterlerinden böler.
            # re.split(r'\s+', line.strip()) komutu, satırı bir veya daha fazla boşluk karakterinden böler.
            parts = re.split(r'\s+', line.strip())
            
            # parts[2] Q mu diye kontrol etmeden önce listenin yeterince uzun olduğundan emin olalım
            if len(parts) > 2 and parts[2] == 'Q':
                total_q_count += 1
                
                # Yandex formatına göre URL'ler 7. sütundan (indeks 6) başlar.
                # Bu sefer listenin geri kalanını sayıyoruz.
                # 6 sabit sütun: SessionID, TimePassed, Q, SERPID, QueryID, ListOfTerms
                num_urls = len(parts) - 6
                
                if num_urls > 10:
                    q_gt_10_count += 1
                elif num_urls < 10:
                    q_lt_10_count += 1
                else:  # num_urls == 10
                    q_eq_10_count += 1

    print("\n--- Analiz Sonuçları ---")
    if total_q_count > 0:
        p_gt_10 = (q_gt_10_count / total_q_count) * 100
        p_lt_10 = (q_lt_10_count / total_q_count) * 100
        p_eq_10 = (q_eq_10_count / total_q_count) * 100
        
        print(f"Toplam Sorgu (Q) Sayısı: {total_q_count:,}")
        print("-" * 30)
        print(f"Kırpılacak Sorgu Sayısı (> 10 ilan): {q_gt_10_count:,} (%{p_gt_10:.2f})")
        print(f"Doldurulacak Sorgu Sayısı (< 10 ilan): {q_lt_10_count:,} (%{p_lt_10:.2f})")
        print(f"Tam Uyan Sorgu Sayısı (== 10 ilan): {q_eq_10_count:,} (%{p_eq_10:.2f})")
        print("-" * 30)
        print("Değerlendirme:")
        print(f"Hedef formata uymak için, sorguların %{p_gt_10:.2f}'sindeki ilan listeleri kırpılacaktır.")
    else:
        print("Dosyada hiç sorgu (Q) satırı bulunamadı.")

def main():
    parser = argparse.ArgumentParser(description="Bir log dosyasındaki sorgu uzunluklarını analiz eder.")
    parser.add_argument('--file', required=True, help="Analiz edilecek log dosyasının yolu.")
    args = parser.parse_args()
    analyze_log_file(args.file)

if __name__ == "__main__":
    main()