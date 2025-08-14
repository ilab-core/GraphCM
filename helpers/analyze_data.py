import pandas as pd
import json
import os 
from collections import defaultdict

def analyze_interactions(data_file_path):
    """
    Veri setindeki (sorgu, doküman) çiftlerinin gösterim ve tıklanma sayılarını analiz eder.
    """
    interactions = []
    
    print(f"{data_file_path} dosyası okunuyor ve analiz ediliyor...")

    # Dosyanın var olup olmadığını kontrol edelim
    if not os.path.exists(data_file_path):
        print(f"HATA: Dosya bulunamadı: {data_file_path}")
        print("Lütfen script'i projenin ana dizininden (GraphCM/) çalıştırdığınızdan emin olun.")
        return None

    with open(data_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            query_id = int(parts[1])
            doc_ids = json.loads(parts[2])
            clicks = json.loads(parts[4])
            
            for doc_id, click in zip(doc_ids, clicks):
                interactions.append({
                    'query_id': query_id,
                    'doc_id': doc_id,
                    'click': click
                })

    if not interactions:
        print("Hiç etkileşim bulunamadı.")
        return None

    # Etkileşimleri bir pandas DataFrame'e dönüştürelim
    df = pd.DataFrame(interactions)
    
    # Her bir (sorgu, doküman) çifti için istatistikleri hesaplayalım
    stats = df.groupby(['query_id', 'doc_id']).agg(
        impression_count=('click', 'count'),  # Kaç kez gösterildiği
        click_count=('click', 'sum')          # Kaç kez tıklandığı
    ).reset_index()
    
    # Tıklama Oranını (CTR) hesaplayalım
    stats['ctr'] = (stats['click_count'] / stats['impression_count']).round(3)
    
    print("\nAnaliz Tamamlandı. İşte en sık gösterilen (sorgu, doküman) çiftleri:")
    
    # En çok gösterilenlere göre sıralayalım
    top_impressions = stats.sort_values(by='impression_count', ascending=False)
    
    print(top_impressions.head(50)) # En popüler ilk 50'yi göster
    
    return top_impressions


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    train_file = os.path.join(PROJECT_ROOT, 'data', 'emj', 'train_per_query_quid.txt')
    
    # Ana analiz fonksiyonunu çalıştırıp tüm istatistikleri alalım
    analysis_results = analyze_interactions(train_file)

    if analysis_results is not None:
        # --- YENİ EKLENEN BÖLÜM ---
        print("\n" + "="*40)
        print("SENARYO C: Orta Popülerlikteki Değerli Örnekler Analizi")
        print("(Gösterim sayısı 50 ile 100 arasında olanlar, CTR'a göre sıralı)")
        print("="*40)

        # Gösterim sayısı 50 ile 100 arasında olanları filtrele
        mid_popularity_pairs = analysis_results[
            (analysis_results['impression_count'] >= 50) & 
            (analysis_results['impression_count'] <= 100)
        ]
        
        # Bu nadir çiftleri CTR'a göre sırala
        valuable_mid_pairs = mid_popularity_pairs.sort_values(by='ctr', ascending=False)
        
        if valuable_mid_pairs.empty:
            print("Bu kriterlere uyan 'Orta Popülerlikte Değerli' örnek bulunamadı.")
        else:
            print(valuable_mid_pairs.head(20).to_string())