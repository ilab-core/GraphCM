import argparse
import os
import re
from collections import defaultdict
from tqdm import tqdm

def find_file_pairs(event_log_dir, session_map_dir):
    """
    İki klasördeki eşleşen günlük dosyaları bulur ve tarih bazında çiftler halinde listeler.
    """
    print(f"-> Adım 0: Eşleşen günlük log dosyaları aranıyor...")
    file_pairs = []
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")

    if not os.path.isdir(event_log_dir):
        print(f"HATA: Olay log klasörü bulunamadı: {event_log_dir}")
        return []
    if not os.path.isdir(session_map_dir):
        print(f"HATA: Oturum haritası klasörü bulunamadı: {session_map_dir}")
        return []

    for event_filename in os.listdir(event_log_dir):
        match = date_pattern.search(event_filename)
        if match:
            date_suffix = match.group(1)
            possible_session_files = [
                f"session_id-ds_search_id_{date_suffix}.csv",
                f"session_id-ds_search_id_{date_suffix}.txt"
            ]
            full_event_path = os.path.join(event_log_dir, event_filename)
            
            for session_file in possible_session_files:
                full_session_path = os.path.join(session_map_dir, session_file)
                if os.path.exists(full_session_path):
                    file_pairs.append((full_event_path, full_session_path))
                    break
                    
    if not file_pairs:
        print("UYARI: Belirtilen klasörlerde eşleşen dosya bulunamadı.")
    else:
        print(f"   {len(file_pairs)} adet eşleşen günlük dosya çifti bulundu.")
        
    return sorted(file_pairs)

def convert_data(event_log_dir, session_map_dir, output_path):
    """
    Belirtilen klasörlerdeki tüm Emlakjet loglarını, Yandex formatında tek bir dosyaya dönüştürür.
    """
    print("Dönüşüm süreci başlatıldı...")

    file_pairs = find_file_pairs(event_log_dir, session_map_dir)
    if not file_pairs:
        return

    # Adım 1: Oturum haritasını (search_id -> session_id) oluştur
    print("-> Adım 1: Oturum haritaları (session maps) birleştiriliyor...")
    search_to_session_map = {}
    for _, session_map_path in tqdm(file_pairs, desc="   Oturum haritaları okunuyor"):
        with open(session_map_path, 'r', encoding='utf-8') as f:
            # Başlık satırını atla (varsa)
            next(f, None) 
            for line in f:
                parts = line.strip().replace('"', '').split(',')
                if len(parts) < 2:
                    parts = line.strip().split()
                if len(parts) >= 2:
                    search_id, session_id = parts[0], parts[1]
                    search_to_session_map[search_id] = session_id
    print(f"   Toplam {len(search_to_session_map)} adet benzersiz search_id -> session_id eşleşmesi bulundu.")

    # Adım 2: Tüm olayları (Q ve C) search_id bazında grupla
    print("-> Adım 2: Olay logları gruplanıyor...")
    events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': []})
    for event_log_path, _ in tqdm(file_pairs, desc="   Olay logları okunuyor   "):
        with open(event_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue

                search_id = parts[0]
                event_type = parts[2]

                if event_type == 'Q' and len(parts) >= 6:
                    events_by_search_id[search_id]['query_info'] = {
                        'term_id': parts[3],
                        'ilan_list': parts[5:]
                    }
                elif event_type == 'C':
                    events_by_search_id[search_id]['clicks'].append(parts[3])

    # Adım 3: Oturumları düzenle ve Yandex formatında yaz
    print(f"-> Adım 3: Yandex formatında çıktı dosyası oluşturuluyor: {output_path}")
    
    sessions_data = defaultdict(list)
    for search_id, data in events_by_search_id.items():
        session_id = search_to_session_map.get(search_id)
        if session_id and data.get('query_info'):
            data['search_id'] = search_id 
            sessions_data[session_id].append(data)

    with open(output_path, 'w', encoding='utf-8') as writer:
        sorted_session_ids = sorted(sessions_data.keys())
        for session_id in tqdm(sorted_session_ids, desc="   Oturumlar yazılıyor     "):
            writer.write(f"{session_id}\tM\t1\t{session_id}\n")
            
            searches_in_session = sessions_data[session_id]
            for search_data in searches_in_session:
                query_info = search_data['query_info']
                serpid = search_data['search_id'] 
                
                query_term_id = query_info['term_id']
                ilan_list = query_info['ilan_list']
                
                url_domain_list = [f"{ilan_id},1" for ilan_id in ilan_list if ilan_id]
                formatted_ilan_string = " ".join(url_domain_list)

                # YENİ KURAL: ListOfTerms alanı için '0' kullanılıyor.
                # SessionID TimePassed TypeOfRecord SERPID QueryID ListOfTerms ListOfURLsAndDomains
                writer.write(f"{session_id}\t0\tQ\t{serpid}\t{query_term_id}\t0\t{formatted_ilan_string}\n")
                
                for clicked_ilan_id in search_data['clicks']:
                    # SessionID TimePassed TypeOfRecord SERPID URLID
                    writer.write(f"{session_id}\t1\tC\t{serpid}\t{clicked_ilan_id}\n")

    print(f"\nDönüşüm tamamlandı. {len(sessions_data)} oturum işlendi. Çıktı dosyası: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Emlakjet'in günlük log klasörlerini Yandex formatına dönüştürür.")
    parser.add_argument('--event-log-dir', required=True, help="Q ve C olay loglarını içeren klasörün yolu.")
    parser.add_argument('--session-map-dir', required=True, help="Search ID -> Session ID eşleşme dosyalarını içeren klasörün yolu.")
    parser.add_argument('--output', default="data/MyDataset/yandex_format_log_FULL.txt", help="Oluşturulacak Yandex formatındaki çıktı dosyasının yolu.")
    
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        print(f"'{output_dir}' klasörü oluşturuluyor...")
        os.makedirs(output_dir)

    convert_data(args.event_log_dir, args.session_map_dir, args.output)

if __name__ == "__main__":
    main()