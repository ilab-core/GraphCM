import re
import sys

# --- LÜTFEN DOSYA YOLLARINI KONTROL EDİN ---
input_file_path = "data/MyDataset/raw_log2.txt"
output_file_path = "data/MyDataset/train_final_and_correct.txt" # Çıktı dosyasına yeni bir isim verelim
# -----------------------------------------

print(f"Kanıta dayalı nihai çözümle işlem başlatılıyor...")
print(f"Girdi: '{input_file_path}'")
print(f"Çıktı: '{output_file_path}'")

try:
    with open(input_file_path, "r", encoding="utf-8") as fin, \
         open(output_file_path, "w", encoding="utf-8") as fout:

        processed_count = 0
        for i, line in enumerate(fin, 1):
            
            # Satırın başındaki ve sonundaki tüm gereksiz boşlukları temizle
            clean_line = line.strip()

            # Boş satırları atla
            if not clean_line:
                continue

            # KANITLANMIŞ YÖNTEM: Satırı, türü ne olursa olsun (boşluk, TAB, vs.)
            # tüm boşluk karakterlerine göre böl.
            parts = re.split(r'\s+', clean_line)
            
            # Bölünen parçaları aralarına tek bir TAB koyarak birleştir ve dosyaya yaz.
            output_line = "\t".join(parts)
            fout.write(output_line + "\n")
            processed_count += 1

    print("\n--- İŞLEM BAŞARIYLA TAMAMLANDI ---")
    print(f"Toplam {processed_count} satır işlendi ve yeni dosyaya yazıldı.")
    print(f"Lütfen '{output_file_path}' dosyasını kontrol edin.")

except FileNotFoundError:
    print(f"HATA: Girdi dosyası bulunamadı! Lütfen dosya yolunu kontrol edin: '{input_file_path}'", file=sys.stderr)
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}", file=sys.stderr)