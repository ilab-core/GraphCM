# -*- coding: utf-8 -*-
import argparse
import os
import re
from collections import defaultdict
from tqdm import tqdm

def load_session_map(session_map_path):
    """Verilen tek bir session map dosyasını okur."""
    print(f"-> Oturum haritası okunuyor: {session_map_path}")
    search_to_session_map = {}
    with open(session_map_path, 'r', encoding='utf-8') as f:
        next(f, None) # Başlık satırını atla
        for line in tqdm(f, desc="   Eşleşmeler okunuyor"):
            parts = [p.strip() for p in line.strip().replace('"', '').split(',')]
            if len(parts) >= 2 and parts[0].isdigit():
                search_id, session_id = parts[0], parts[1]
                search_to_session_map[search_id] = session_id
    print(f"   Toplam {len(search_to_session_map):,} adet benzersiz search_id -> session_id eşleşmesi bulundu.")
    return search_to_session_map

def convert_to_yandex_format(event_log_path, search_to_session_map, output_path, max_listings=30):
    """Verilen tek bir event log dosyasını okur ve 30 ilana göre padding/truncate yaparak nihai Yandex formatına dönüştürür."""
    
    print("\n" + "="*50)
    print(f"İŞLEM BAŞLADI: '{os.path.basename(event_log_path)}' dosyası işleniyor...")

    print("\n-> Adım 1: Olay logları okunup gruplanıyor...")
    events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': []})
    with open(event_log_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="   Olay logları okunuyor"):
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            search_id, event_type = parts[0], parts[2]
            
            if event_type == 'Q' and len(parts) >= 5:
                events_by_search_id[search_id]['query_info'] = parts
            elif event_type == 'C':
                if len(parts) >= 4:
                    events_by_search_id[search_id]['clicks'].append(parts[3])

    print(f"\n-> Adım 2: Yandex formatında çıktı dosyası oluşturuluyor: '{output_path}'")
    sessions_output = defaultdict(list)
    
    for search_id, data in tqdm(events_by_search_id.items(), desc="   Oturumlar işleniyor"):
        session_id = search_to_session_map.get(search_id)
        query_parts = data.get('query_info')
        
        if session_id and query_parts:
            time_passed = query_parts[1]
            serpid = query_parts[0]         # SERPID, ham verideki search_id'dir.
            query_id = query_parts[3]       # QueryID, ham verideki 4. sütundur.
            list_of_terms = query_parts[4]  # ListOfTerms, ham verideki 5. sütundur.
            doc_list_raw = query_parts[5:]    # İlanlar 6. sütundan başlar.

            if len(doc_list_raw) > max_listings:
                processed_docs_raw = doc_list_raw[:max_listings]  
            else:     
                processed_docs_raw = doc_list_raw + ['0'] * (max_listings - len(doc_list_raw))
            
            formatted_docs = [f"{doc_id},1" for doc_id in processed_docs_raw]
            processed_doc_ids = {doc_id for doc_id in processed_docs_raw if doc_id != '0'}

            q_line_elements = [session_id, time_passed, 'Q', serpid, query_id, list_of_terms] + formatted_docs
            
            c_lines = []
            for clicked_doc_id in data['clicks']:
                if clicked_doc_id in processed_doc_ids:
                    # C Satırı: SessionID TimePassed TypeOfRecord SERPID URLID
                    # Not: C satırı için TimePassed genellikle '1' olarak ayarlanır.
                    c_line_parts = [session_id, '1', 'C', serpid, clicked_doc_id]
                    c_lines.append("\t".join(c_line_parts))
            
            sessions_output[session_id].append({
                "m_line": f"{session_id}\tM\t1\t{session_id}",
                "q_line": "\t".join(q_line_elements),
                "c_lines": c_lines
            })

    with open(output_path, 'w', encoding='utf-8') as writer:
        sorted_session_ids = sorted(sessions_output.keys())
        for session_id in tqdm(sorted_session_ids, desc="   Oturumlar yazılıyor"):
            if sessions_output[session_id]:
                writer.write(sessions_output[session_id][0]['m_line'] + "\n")
            
            for search_event in sessions_output[session_id]:
                writer.write(search_event['q_line'] + "\n")
                for c_line in search_event['c_lines']:
                    writer.write(c_line + "\n")
                    
    print(f"\n✅ İşlem tamamlandı. {len(sessions_output)} oturum işlendi.")
    print(f"   Çıktı dosyası: '{output_path}'")

def main():
    parser = argparse.ArgumentParser(description="Ham log dosyalarını 30 ilana göre padding yaparak nihai Yandex formatına dönüştürür.")
    
    parser.add_argument('--map-file', required=True, help="Train ve Test verilerini içeren ortak session harita dosyası.")
    parser.add_argument('--train-log', required=True, help="Train setine ait ham olay log dosyası (örn: train.txt).")
    parser.add_argument('--train-output', default="train_final.txt", help="Oluşturulacak nihai train dosyası.")
    parser.add_argument('--test-log', required=True, help="Test setine ait ham olay log dosyası (örn: test.txt).")
    parser.add_argument('--test-output', default="test_final.txt", help="Oluşturulacak nihai test dosyası.")
    
    args = parser.parse_args()

    session_map = load_session_map(args.map_file)
    if not session_map:
        print("HATA: Oturum haritası yüklenemediği için işlem durduruldu.")
        return

    if args.train_log:
        output_dir = os.path.dirname(args.train_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        convert_to_yandex_format(args.train_log, session_map, args.train_output)

    if args.test_log:
        output_dir = os.path.dirname(args.test_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        convert_to_yandex_format(args.test_log, session_map, args.test_output)

if __name__ == "__main__":
    main()

