import argparse
import os
import re
from tqdm import tqdm

def check_file_integrity(filepath):
    """
    Nihai dosyayı analiz eder ve formatının doğruluğunu kontrol eder.
    """
    print(f"\n--- '{os.path.basename(filepath)}' Dosyası Kontrol Ediliyor ---")

    if not os.path.exists(filepath):
        print(f"HATA: Belirtilen yolda dosya bulunamadı: {filepath}")
        return

    # Sayaclar ve kontrol degiskenleri
    p_row_count = 0
    q_row_count = 0
    q_with_30_listings = 0
    q_with_other_listings = 0
    unique_sessions = set()
    first_50_lines = []

    try:
        # TQDM için toplam satır sayısını al
        with open(filepath, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        print(f"-> Dosya içeriği taranıyor...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="   İşleniyor")):
                if i < 50:
                    first_50_lines.append(line.strip())

                line = line.strip()
                if not line:
                    continue
                
                parts = re.split(r'\s+', line)
                
                # Benzersiz session ID'sini sete ekle
                unique_sessions.add(parts[0])

                # Satır tipini kontrol et
                if len(parts) > 2:
                    row_type = parts[2]
                    if row_type == 'P':
                        p_row_count += 1
                    elif row_type == 'Q':
                        q_row_count += 1
                        # İlan sayısını kontrol et (5 meta sütunu sonrası)
                        num_listings = len(parts) - 5
                        if num_listings == 30:
                            q_with_30_listings += 1
                        else:
                            q_with_other_listings += 1
    
    except Exception as e:
        print(f"HATA: Dosya okunurken bir sorun oluştu: {e}")
        return

    print("\n--- KONTROL RAPORU ---")
    print(f"Dosya Adı: {os.path.basename(filepath)}")
    print("-" * 30)
    
    # 1. 'P' satırı kontrolü
    print(f"[Kontrol 1: 'P' Satırları]")
    print(f"  Bulunan 'P' satırı sayısı: {p_row_count}")
    if p_row_count == 0:
        print("  ✅ BAŞARILI: 'P' satırları başarıyla kaldırılmış.")
    else:
        print("  ❌ BAŞARISIZ: Dosyada hala 'P' satırları bulunuyor.")
    
    # 2. Benzersiz Oturum Sayısı
    print(f"\n[Kontrol 2: Benzersiz Oturum Sayısı]")
    print(f"  Bulunan benzersiz session ID sayısı: {len(unique_sessions):,}")
    print("  ℹ️ Bu sayının, bir önceki script'in raporladığı 'güncellenen oturum' sayısıyla eşleşmesi beklenir.")

    # 3. İlan Sayısı Kontrolü
    print(f"\n[Kontrol 3: 'Q' Satırlarındaki İlan Sayısı]")
    print(f"  Toplam 'Q' satırı sayısı: {q_row_count:,}")
    print(f"  Tam olarak 30 ilan içeren 'Q' satırı: {q_with_30_listings:,}")
    print(f"  Farklı sayıda ilan içeren 'Q' satırı: {q_with_other_listings:,}")
    if q_with_other_listings == 0 and q_row_count > 0:
        print("  ✅ BAŞARILI: Tüm 'Q' satırları 30 ilan içermektedir.")
    elif q_row_count > 0:
        print("  ❌ BAŞARISIZ: Bazı 'Q' satırları 30 ilan içermiyor.")

    print("-" * 30)

    # 4. Manuel Kontrol için İlk 50 Satır
    print("\n--- İlk 50 Satır (Manuel Kontrol İçin) ---")
    if first_50_lines:
        for line in first_50_lines:
            print(line)
    else:
        print("Dosya boş veya okunamadı.")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Nihai Yandex formatındaki dosyanın bütünlüğünü kontrol eder."
    )
    parser.add_argument('--file', required=True, help="Kontrol edilecek dosyanın yolu (örn: train_final.txt).")
    
    args = parser.parse_args()
    check_file_integrity(args.file)

if __name__ == "__main__":
    main()
