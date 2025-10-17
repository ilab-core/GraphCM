# check_raw_click_overlap.py (Örnek Çıktı Eklenmiş Versiyon)
import os
from tqdm import tqdm
from collections import defaultdict

def analyze_raw_click_overlap(train_path, test_path):
    """
    Ham train.txt ve test.txt dosyalarını okur. Train setinde tıklanmış olan 
    (sorgu, ilan) çiftlerinin, test setinde tıklanmış olan çiftlerle ne 
    kadar kesiştiğini analiz eder.
    """
    print("="*60)
    print("HAM VERİDE Tıklanmış Olay Kesişimi Analizi Başlatıldı...")
    print("="*60)

    def get_clicked_pairs_from_raw_file(file_path, desc):
        """Yardımcı fonksiyon: Ham dosyadan tıklanmış (sorgu, ilan) çiftlerini çıkarır."""
        
        events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': set()})
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split() # Boşluklara göre ayır
                    event_type = parts[2]
                    search_id = parts[0]
                    
                    if event_type == 'Q' and len(parts) >= 5:
                        events_by_search_id[search_id]['query_info'] = parts
                    elif event_type == 'C' and len(parts) >= 4:
                        events_by_search_id[search_id]['clicks'].add(parts[3])
                except IndexError:
                    continue

        clicked_pairs = set()
        for search_id, data in tqdm(events_by_search_id.items(), desc=desc):
            query_info = data.get('query_info')
            clicks = data.get('clicks')

            if query_info and clicks:
                query_id = query_info[3]
                doc_ids = query_info[5:]
                
                for doc_id in doc_ids:
                    if doc_id in clicks:
                        clicked_pairs.add((query_id, doc_id))
                        
        return clicked_pairs

    # Adım 1: Train setindeki tıklanmış çiftleri bul
    print(f"\n[Adım 1] '{train_path}' dosyasındaki tıklanmış olaylar okunuyor...")
    train_clicked_pairs = get_clicked_pairs_from_raw_file(train_path, "--> Train dosyası işleniyor")
    print(f"--> Train setinde {len(train_clicked_pairs):,} adet benzersiz tıklanmış (sorgu, ilan) çifti bulundu.")

    # Adım 2: Test setindeki tıklanmış çiftleri bul
    print(f"\n[Adım 2] '{test_path}' dosyasındaki tıklanmış olaylar analiz ediliyor...")
    test_clicked_pairs = get_clicked_pairs_from_raw_file(test_path, "--> Test dosyası işleniyor")
    
    # Adım 3: Kesişimi hesapla ve sonuçları raporla
    print("\n" + "="*60)
    print("ANALİZ SONUÇLARI")
    print("="*60)
    
    if not test_clicked_pairs:
        print("\n-> Test setinde hiç tıklanmış olay bulunamadı. Analiz yapılamıyor.")
    else:
        overlap_pairs = train_clicked_pairs.intersection(test_clicked_pairs)
        overlap_count = len(overlap_pairs)
        overlap_ratio = (overlap_count / len(test_clicked_pairs)) * 100

        print(f"\n-> Train setindeki benzersiz tıklanmış çift sayısı: {len(train_clicked_pairs):,}")
        print(f"-> Test setindeki benzersiz tıklanmış çift sayısı:  {len(test_clicked_pairs):,}")
        print(f"-> Bu iki kümenin kesişimindeki çift sayısı:      {overlap_count:,}")
        print(f"\n-> Kesişim Oranı (Cevap Anahtarı Sızıntısı): %{overlap_ratio:.2f}")

        print("\n" + "-"*60)
        if overlap_ratio > 10:
            print("🚨 KESİN KANIT! Test setindeki tıklamaların önemli bir kısmı Train setinde de mevcut.")
            print("   Bu, 'test_ppl=1.0' sonucunu ve modelin ezber yaptığını %100 doğrulamaktadır.")

            # --- YENİ EKLENEN KISIM: Ortak çiftlerden 50 örnek yazdır ---
            if overlap_pairs:
                print("\n--- Kesişen (Sızan) Çiftlerden Örnekler (İlk 50) ---")
                print("    (Sorgu ID, İlan ID)")
                print("    --------------------------------------------")
                
                for i, pair in enumerate(list(overlap_pairs)[:50]):
                    query_id, doc_id = pair
                    print(f"    {i+1:2d}. ({query_id}, {doc_id})")
            # --- YENİ KISIM SONU ---

        else:
            print("✅ Düşük Kesişim. Basit bir cevap anahtarı sızıntısı görünmüyor.")
    
    print("="*60)


if __name__ == '__main__':
    TRAIN_PATH = 'raw_data/train.txt'
    TEST_PATH = 'raw_data/test.txt'
    analyze_raw_click_overlap(TRAIN_PATH, TEST_PATH)