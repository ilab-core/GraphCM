import os

def debug_file_format(filepath):
    """
    Verilen dosyanın ilk 10 satırını okur ve her satırın yapısını
    (sütun sayısı ve içeriği) analiz eder.
    """
    print(f"\n--- '{os.path.basename(filepath)}' Dosya Formatı İnceleniyor ---")

    if not os.path.exists(filepath):
        print(f"HATA: Dosya bulunamadı: {filepath}")
        return

    print("Dosyanın ilk 10 satırı ve sütun sayıları:\n")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                
                line = line.strip()
                # Satırı tab karakterine göre bölüyoruz
                parts = line.split('\t')
                num_parts = len(parts)
                
                print(f"Satır {i+1}:")
                print(f"  -> Sütun Sayısı: {num_parts}")
                print(f"  -> İçerik: {parts}\n")
                
                if num_parts != 5:
                    print(f"  --> UYARI: Bu satırın sütun sayısı 5 DEĞİL! analyze_data.py bu satırı atlayacaktır.\n")

    except Exception as e:
        print(f"HATA: Dosya okunurken bir sorun oluştu: {e}")

if __name__ == "__main__":
    # Analiz edilecek dosyanın yolu
    target_file = 'data/emj_30ilan/train_per_query_quid.txt'
    debug_file_format(target_file)