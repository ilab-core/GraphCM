# check_click_overlap.py (DÜZELTİLMİŞ VERSİYON)
import os
from tqdm import tqdm
from collections import defaultdict

def analyze_click_overlap(train_path, test_path):
    """
    Train setinde tıklanmış olan (sorgu, ilan) çiftlerinin, test setinde 
    tıklanmış olan çiftlerle ne kadar kesiştiğini analiz eder.
    """
    print("="*60)
    print("Tıklanmış Olay Kesişimi (Click Overlap) Analizi Başlatıldı...")
    print("="*60)

    # --- YENİ VE DÜZELTİLMİŞ MANTIK ---
    def get_clicked_pairs_from_file(file_path, desc):
        """Yardımcı fonksiyon: Bir dosyadan tıklanmış (sorgu, ilan) çiftlerini çıkarır."""
        
        # Önce tüm satırları oturumlara göre gruplayalım
        sessions = defaultdict(list)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sessions[line.strip().split('\t')[0]].append(line.strip())

        clicked_pairs = set()
        total_clicks = 0

        for session_id, lines in tqdm(sessions.items(), desc=desc):
            # Oturum içindeki Q ve C olaylarını ayır
            q_events = [parts for parts in (line.split('\t') for line in lines) if len(parts) > 2 and parts[2] == 'Q']
            c_events = [parts for parts in (line.split('\t') for line in lines) if len(parts) > 2 and parts[2] == 'C']

            for q_parts in q_events:
                if len(q_parts) < 7: continue # Bozuk Q satırını atla
                
                current_serpid = q_parts[3]
                current_query_id = q_parts[4]
                
                # HATA DÜZELTME: Tıklamaları SERPID ile doğru sorguya bağlıyoruz.
                clicks_for_this_query = {c_parts[4] for c_parts in c_events if len(c_parts) > 4 and c_parts[3] == current_serpid}
                
                if not clicks_for_this_query: continue

                doc_ids = [ud.split(',')[0] for ud in q_parts[6:]]
                for doc_id in doc_ids:
                    if doc_id in clicks_for_this_query:
                        clicked_pairs.add((current_query_id, doc_id))
                        total_clicks += 1
                        
        return clicked_pairs, total_clicks
    # --- YENİ MANTIK SONU ---


    # Adım 1: Train setindeki tıklanmış çiftleri bul
    print(f"\n[Adım 1] '{train_path}' dosyasındaki tıklanmış olaylar okunuyor...")
    train_clicked_pairs, _ = get_clicked_pairs_from_file(train_path, "--> Train oturumları işleniyor")
    print(f"--> Train setinde {len(train_clicked_pairs):,} adet benzersiz tıklanmış (sorgu, ilan) çifti bulundu.")

    # Adım 2: Test setindeki tıklanmış çiftleri ve kesişimi bul
    print(f"\n[Adım 2] '{test_path}' dosyasındaki tıklanmış olaylar analiz ediliyor...")
    test_clicked_pairs, test_total_clicks = get_clicked_pairs_from_file(test_path, "--> Test oturumları işleniyor")
    
    overlap_count = 0
    for test_pair in test_clicked_pairs:
        if test_pair in train_clicked_pairs:
            overlap_count += 1
    
    # Adım 3: Sonuçları raporla
    print("\n" + "="*60)
    print("ANALİZ SONUÇLARI")
    print("="*60)
    
    if test_total_clicks == 0:
        print("\n-> Test setinde hiç tıklanmış olay bulunamadı. Analiz yapılamıyor.")
    else:
        # Kesişimi, benzersiz çiftler üzerinden hesaplayalım
        overlap_ratio = (overlap_count / len(test_clicked_pairs)) * 100 if len(test_clicked_pairs) > 0 else 0
        print(f"\n-> Train setindeki benzersiz tıklanmış çift sayısı: {len(train_clicked_pairs):,}")
        print(f"-> Test setindeki benzersiz tıklanmış çift sayısı:  {len(test_clicked_pairs):,}")
        print(f"-> Bu iki kümenin kesişimindeki çift sayısı:      {overlap_count:,}")
        print(f"\n-> Kesişim Oranı (Benzersiz Çiftler Üzerinden): %{overlap_ratio:.2f}")

        print("\n" + "-"*60)
        if overlap_ratio > 10:
            print("🚨 Yüksek Kesişim! Bu oran, gözlemlediğimiz aşırı öğrenmenin (test_ppl=1.0)")
            print("   nedenini güçlü bir şekilde açıklıyor. Model, test setindeki cevapları train setinde ezberlemiş.")
        else:
            print("✅ Düşük Kesişim. Basit bir cevap anahtarı sızıntısı görünmüyor.")
    
    print("="*60)


if __name__ == '__main__':
    TRAIN_FILE_PATH = 'raw_data/train_final.txt'
    TEST_FILE_PATH = 'raw_data/test_final.txt'
    analyze_click_overlap(TRAIN_FILE_PATH, TEST_FILE_PATH)