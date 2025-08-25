# -*- coding: utf-8 -*-
import argparse
import os
import re
from collections import defaultdict
from tqdm import tqdm

def find_file_pairs(event_log_dir, session_map_dir):
    """İki klasördeki eşleşen günlük dosyaları bulur."""
    print("-> Adım 0: Eşleşen günlük log dosyaları aranıyor...")
    file_pairs = []
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
    if not os.path.isdir(event_log_dir) or not os.path.isdir(session_map_dir):
        print(f"HATA: Girdi klasörleri bulunamadı.")
        return []

    for event_filename in os.listdir(event_log_dir):
        match = date_pattern.search(event_filename)
        if match:
            date_suffix = match.group(1)
            possible_session_files = [f"session_id-ds_search_id_{date_suffix}.csv", f"session_id-ds_search_id_{date_suffix}.txt"]
            full_event_path = os.path.join(event_log_dir, event_filename)
            for session_file in possible_session_files:
                full_session_path = os.path.join(session_map_dir, session_file)
                if os.path.exists(full_session_path):
                    file_pairs.append((full_event_path, full_session_path))
                    break
    print(f"   {len(file_pairs)} adet eşleşen günlük dosya çifti bulundu.")
    return sorted(file_pairs)

def convert_to_yandex_format(event_log_dir, session_map_dir, output_path):
    """Ham logları okur ve doğru Yandex formatında tek bir dosyaya yazar."""
    file_pairs = find_file_pairs(event_log_dir, session_map_dir)
    if not file_pairs: return

    print("\n-> Adım 1: Oturum haritaları birleştiriliyor...")
    search_to_session_map = {}
    for _, session_map_path in tqdm(file_pairs, desc="   Oturum haritaları okunuyor"):
        with open(session_map_path, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
                if len(parts) >= 2 and parts[0].isdigit():
                    search_id, session_id = parts[0], parts[1]
                    search_to_session_map[search_id] = session_id
    print(f"   Toplam {len(search_to_session_map)} adet benzersiz search_id -> session_id eşleşmesi bulundu.")

    print("\n-> Adım 2: Olay logları okunup gruplanıyor...")
    events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': []})
    for event_log_path, _ in tqdm(file_pairs, desc="   Olay logları okunuyor     "):
        with open(event_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 4: continue
                search_id, event_type = parts[0], parts[2]
                if event_type == 'Q' and len(parts) >= 6:
                    events_by_search_id[search_id]['query_info'] = parts
                elif event_type == 'C':
                    # P olaylarını (dönüşüm) görmezden geliyoruz
                    events_by_search_id[search_id]['clicks'].append(parts[3])

    print(f"\n-> Adım 3: Yandex formatında çıktı dosyası oluşturuluyor...")
    sessions_output = defaultdict(list)
    
    for search_id, data in events_by_search_id.items():
        session_id = search_to_session_map.get(search_id)
        query_parts = data.get('query_info')
        
        if session_id and query_parts:
            doc_list_raw = query_parts[5:]
            if len(doc_list_raw) >= 10:
                top_10_docs_raw = doc_list_raw[:10]
                top_10_docs_with_domain = [f"{doc_id},1" for doc_id in top_10_docs_raw]
                top_10_doc_ids_only = {doc_id for doc_id in top_10_docs_raw}
                
                # --- YENİ DÜZELTİLMİŞ BÖLÜM ---
                # Yandex formatı için Q ve C satırlarını doğru sırada hazırla
                time_passed = query_parts[1]
                serpid = query_parts[0]
                query_id = query_parts[3]
                list_of_terms = query_parts[4]

                # Tüm sütunları doğru sırayla bir listeye koy
                q_line_elements = [
                    session_id, time_passed, 'Q', serpid, query_id, list_of_terms
                ] + top_10_docs_with_domain
                
                c_lines = []
                for clicked_doc_id in data['clicks']:
                    if clicked_doc_id in top_10_doc_ids_only:
                        c_line_parts = [session_id, '1', 'C', serpid, clicked_doc_id]
                        c_lines.append("\t".join(c_line_parts))
                
                sessions_output[session_id].append({
                    "m_line": f"{session_id}\tM\t1\t{session_id}",
                    "q_line": "\t".join(q_line_elements),
                    "c_lines": c_lines
                })
                # --- DÜZELTME SONU ---

    with open(output_path, 'w', encoding='utf-8') as writer:
        sorted_session_ids = sorted(sessions_output.keys())
        for session_id in tqdm(sorted_session_ids, desc="   Oturumlar yazılıyor      "):
            # Her oturum için sadece bir M satırı yazıldığından emin ol
            if sessions_output[session_id]:
                 writer.write(sessions_output[session_id][0]['m_line'] + "\n")
            
            for search_event in sessions_output[session_id]:
                writer.write(search_event['q_line'] + "\n")
                for c_line in search_event['c_lines']:
                    writer.write(c_line + "\n")
                    
    print(f"\nİşlem tamamlandı. {len(sessions_output)} oturum işlendi.")
    print(f"Çıktı dosyası: '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Ham log dosyalarını Yandex formatına dönüştürür.")
    parser.add_argument('--event-log-dir', required=True, help="Q ve C olay loglarını içeren klasörün yolu.")
    parser.add_argument('--session-map-dir', required=True, help="Search ID -> Session ID eşleşme dosyalarını içeren klasörün yolu.")
    parser.add_argument('--output', default="data/train.txt", help="Oluşturulacak Yandex formatındaki çıktı dosyasının yolu.")
    
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    convert_to_yandex_format(args.event_log_dir, args.session_map_dir, args.output)

if __name__ == "__main__":
    main()