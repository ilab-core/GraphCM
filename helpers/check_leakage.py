# check_leakage.py
import os
from tqdm import tqdm

def check_session_leakage(train_path, test_path):
    """
    Train ve test setleri arasında ortak session_id olup olmadığını kontrol ederek
    veri sızıntısını tespit eder ve ortak ID'leri bir dosyaya yazar.
    """
    print("="*50)
    print("Veri Sızıntısı Kontrolü Başlatıldı...")
    print(f"Train Dosyası: {train_path}")
    print(f"Test Dosyası:  {test_path}")
    print("="*50)

    # Adım 1: Train setindeki tüm session ID'leri hızlı arama için bir set'e yükle.
    print("\n[Adım 1] Train setindeki tüm session ID'ler okunuyor...")
    train_session_ids = set()
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="--> Train dosyası taranıyor"):
                try:
                    # Her satırın ilk sütunu session_id'dir.
                    session_id = line.strip().split('\t')[0]
                    train_session_ids.add(session_id)
                except IndexError:
                    # Boş veya bozuk satırları atla
                    continue
    except FileNotFoundError:
        print(f"\nHATA: Train dosyası bulunamadı: {train_path}")
        return

    print(f"--> Train setinde {len(train_session_ids):,} adet benzersiz session ID bulundu.")

    # Adım 2: Test setindeki her bir session ID'nin train setinde olup olmadığını kontrol et.
    print("\n[Adım 2] Test setindeki session ID'ler, train setiyle karşılaştırılıyor...")
    overlapping_sessions = set()
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="--> Test dosyası taranıyor"):
                try:
                    session_id = line.strip().split('\t')[0]
                    if session_id in train_session_ids:
                        overlapping_sessions.add(session_id)
                except IndexError:
                    continue
    except FileNotFoundError:
        print(f"\nHATA: Test dosyası bulunamadı: {test_path}")
        return

    # Adım 3: Sonuçları raporla ve dosyaya kaydet.
    print("\n" + "="*50)
    print("KONTROL SONUÇLARI")
    print("="*50)
    if not overlapping_sessions:
        print("\n✅ TEBRİKLER! Veri Sızıntısı Bulunmadı.")
        print("   Train ve test setleri arasında herhangi bir ortak session ID tespit edilmedi.")
    else:
        print(f"\n🚨 DİKKAT! Veri Sızıntısı Tespit Edildi!")
        print(f"   Train ve test setleri arasında {len(overlapping_sessions):,} adet ortak session ID bulundu.")
        
        # --- GÜNCELLENEN KISIM: Sızan ID'leri dosyaya yazma ---
        output_file = 'leaking_sessions.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            # Tutarlı bir çıktı için ID'leri sıralayarak yazalım.
            sorted_leaks = sorted(list(overlapping_sessions))
            for session_id in sorted_leaks:
                f.write(f"{session_id}\n")
        print(f"\n   -> Sızıntıya neden olan {len(overlapping_sessions)} ID, '{output_file}' dosyasına kaydedildi.")
        # --- GÜNCELLENEN KISIM SONU ---

        print("\n   Sızıntıya neden olan bazı ortak session ID'ler:")
        sample_size = 10
        for i, session_id in enumerate(list(overlapping_sessions)[:sample_size]):
            print(f"   - {session_id}")
        
        if len(overlapping_sessions) > sample_size:
            print(f"   ...ve {len(overlapping_sessions) - sample_size} diğerleri.")
    print("\n" + "="*50)


if __name__ == '__main__':
    # Lütfen bu dosya yollarının kendi projenle uyumlu olduğundan emin ol.
    TRAIN_FILE_PATH = 'raw_data/train_final.txt'
    TEST_FILE_PATH = 'raw_data/test_final.txt'

    # Script'i çalıştırmadan önce dosyaların varlığını kontrol edelim.
    if not os.path.exists(TRAIN_FILE_PATH) or not os.path.exists(TEST_FILE_PATH):
        print("\nHATA: Dosya yolları bulunamadı. Lütfen script içerisindeki")
        print("TRAIN_FILE_PATH ve TEST_FILE_PATH değişkenlerini kendi dosya yollarınızla güncelleyin.")
        print(f"Beklenen train dosyası: {os.path.abspath(TRAIN_FILE_PATH)}")
        print(f"Beklenen test dosyası:  {os.path.abspath(TEST_FILE_PATH)}")
    else:
        check_session_leakage(TRAIN_FILE_PATH, TEST_FILE_PATH)