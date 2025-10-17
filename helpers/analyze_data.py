# -*- coding: utf-8 -*-

import pandas as pd
import json
import os
import sys
import argparse
from tqdm import tqdm

def analyze_interactions(data_file_path):
    """
    Veri setindeki (sorgu, doküman) çiftlerinin gösterim, tıklanma sayılarını
    ve pozisyonlarını analiz eder. 5 sütunlu ve padding ID'si 0 olan formata göre güncellenmiştir.
    """
    interactions = []
    
    print(f"\n'{os.path.basename(data_file_path)}' dosyası okunuyor ve analiz ediliyor...")

    if not os.path.exists(data_file_path):
        print(f"HATA: Dosya bulunamadı: {data_file_path}")
        return None, None

    with open(data_file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="   Satırlar işleniyor"):
            try:
                parts = line.strip().split('\t')
                # DÜZELTME: Sütun sayısı kontrolü 5 olarak güncellendi.
                if len(parts) != 5:
                    continue
                
                query_id = int(parts[1])
                doc_ids = json.loads(parts[2])
                clicks = json.loads(parts[4])
                
                for i, (doc_id, click) in enumerate(zip(doc_ids, clicks)):
                    # DÜZELTME: Padding ID'sinin 0 olduğunu varsayıyoruz.
                    if doc_id == 0:
                        continue
                    
                    interactions.append({
                        'query_id': query_id,
                        'doc_id': doc_id,
                        'click': click,
                        'position': i + 1
                    })
            except (json.JSONDecodeError, IndexError, ValueError) as e:
                print(f"UYARI: Hatalı formatlı satır atlandı: {line.strip()} - Hata: {e}")
                continue

    if not interactions:
        print("Hiç geçerli etkileşim bulunamadı.")
        return None, None

    # Etkileşimleri bir pandas DataFrame'e dönüştürelim
    full_df = pd.DataFrame(interactions)
    
    # Her bir (sorgu, doküman) çifti için genel istatistikleri hesaplayalım
    print("\n-> İstatistikler hesaplanıyor...")
    stats_df = full_df.groupby(['query_id', 'doc_id']).agg(
        impression_count=('click', 'count'),
        click_count=('click', 'sum')
    ).reset_index()
    
    stats_df['ctr'] = (stats_df['click_count'] / stats_df['impression_count']).round(4)
    
    print("Analiz Tamamlandı.")
    
    return stats_df, full_df

def main():
    parser = argparse.ArgumentParser(description="GraphCM veri setindeki etkileşimleri analiz eder.")
    parser.add_argument('set_to_analyze', choices=['train', 'test'], help="Analiz edilecek set (train veya test).")
    parser.add_argument('--data-dir', default='data/emj_30ilan', help="İşlenmiş verilerin bulunduğu klasör.")
    
    args = parser.parse_args()
    
    data_file = os.path.join(args.data_dir, f'{args.set_to_analyze}_per_query_quid.txt')
    
    analysis_results, all_interactions_df = analyze_interactions(data_file)

    if analysis_results is not None:
        # --- BÖLÜM 1: En Popüler Çiftler ---
        print("\n" + "="*50)
        print("BÖLÜM 1: En Popüler (En Sık Gösterilen) 50 Çift")
        print("="*50)
        top_impressions = analysis_results.sort_values(by='impression_count', ascending=False)
        print(top_impressions.head(50).to_string())

        # --- BÖLÜM 2: Orta Popülerlikteki Değerli Örnekler ---
        print("\n" + "="*50)
        print("BÖLÜM 2: Orta Popülerlikteki Değerli Örnekler (50-100 Gösterim)")
        print("="*50)
        mid_popularity_pairs = analysis_results[
            (analysis_results['impression_count'] >= 50) & 
            (analysis_results['impression_count'] <= 100)
        ]
        valuable_mid_pairs = mid_popularity_pairs.sort_values(by='ctr', ascending=False)
        
        if valuable_mid_pairs.empty:
            print("Bu kriterlere uyan örnek bulunamadı.")
        else:
            print(valuable_mid_pairs.head(20).to_string())
        
        # --- BÖLÜM 3: Genel ve Pozisyon 1 İstatistikleri ---
        print("\n" + "="*50)
        print("BÖLÜM 3: Genel ve Pozisyon 1 Ortalama CTR'ları")
        print("="*50)
        
        total_impressions = all_interactions_df.shape[0]
        total_clicks = all_interactions_df['click'].sum()
        overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        
        print(f"Genel İstatistikler:")
        print(f"  - Toplam Gerçek Gösterim: {total_impressions:,}")
        print(f"  - Toplam Tıklanma: {total_clicks:,}")
        print(f"  - Genel Ortalama CTR: {overall_ctr:.4f}")
        
        pos1_df = all_interactions_df[all_interactions_df['position'] == 1]
        if not pos1_df.empty:
            pos1_clicks = pos1_df['click'].sum()
            pos1_impressions = len(pos1_df)
            pos1_ctr = pos1_clicks / pos1_impressions
            print(f"\nPozisyon 1 İstatistikleri:")
            print(f"  - Pozisyon 1 Gösterimleri: {pos1_impressions:,}")
            print(f"  - Pozisyon 1 Tıklamaları: {pos1_clicks:,}")
            print(f"  - Pozisyon 1 Ortalama CTR: {pos1_ctr:.4f}")

if __name__ == "__main__":
    main()

