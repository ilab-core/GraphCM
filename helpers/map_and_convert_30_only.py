# helpers/map_and_convert_30_only.py
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

def convert_to_yandex_format_30_only(event_log_path, search_to_session_map, output_path):
    """
    Verilen event log dosyasını okur, SADECE 30 ilana sahip sorguları filtreler
    ve nihai Yandex formatına dönüştürür. (Padding/Truncate YOK)
    """
    
    print("\n" + "="*50)
    print(f"İŞLEM BAŞLADI: '{os.path.basename(event_log_path)}' dosyası işleniyor...")

    print("\n-> Adım 1: Olay logları okunup gruplanıyor...")
    events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': []})
    with open(event_log_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="   Olay logları okunuyor"):
            # Ham verideki boşluklar düzensiz olabildiği için re.split kullanmak daha güvenli
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 3: continue
            
            search_id, event_type = parts[0], parts[2]
            
            if event_type == 'Q' and len(parts) >= 5:
                events_by_search_id[search_id]['query_info'] = parts
            elif event_type == 'C' and len(parts) >= 4:
                # Ham formatta tıklanan ilan ID'si 4. sütundadır (indeks 3)
                events_by_search_id[search_id]['clicks'].append(parts[3])

    print(f"\n-> Adım 2: Sadece 30 ilan içeren sorgular filtreleniyor ve Yandex formatı oluşturuluyor...")
    sessions_output = defaultdict(list)
    
    for search_id, data in tqdm(events_by_search_id.items(), desc="   Oturumlar işleniyor"):
        session_id = search_to_session_map.get(search_id)
        query_parts = data.get('query_info')
        
        if session_id and query_parts:
            # Ham formatta: Q -> search_id, 0, Q, query_id, 0, doc1, doc2...
            # Bu yüzden ilanlar 6. elemandan (indeks 5) başlar.
            doc_list_raw = query_parts[5:]
            
            # --- ANA DEĞİŞİKLİK BURADA ---
            # Eğer ilan listesi tam olarak 30 elemanlı değilse, bu sorguyu atla.
            if len(doc_list_raw) != 30:
                continue
            # --- DEĞİŞİKLİK SONU ---

            # Artık padding veya kırpma yok, liste zaten 30 elemanlı.
            processed_docs_raw = doc_list_raw
            
            # Kalan mantık tamamen aynı
            time_passed = query_parts[1]
            serpid = query_parts[0]
            query_id = query_parts[3]
            list_of_terms = query_parts[4]
            
            formatted_docs = [f"{doc_id},1" for doc_id in processed_docs_raw]
            processed_doc_ids = set(processed_docs_raw)

            q_line_elements = [session_id, time_passed, 'Q', serpid, query_id, list_of_terms] + formatted_docs
            
            c_lines = []
            for clicked_doc_id in data['clicks']:
                if clicked_doc_id in processed_doc_ids:
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
    parser = argparse.ArgumentParser(description="Ham log dosyalarından sadece 30 ilan içerenleri filtreleyerek Yandex formatına dönüştürür.")
    
    parser.add_argument('--map-file', required=True, help="Ortak session harita dosyası.")
    parser.add_argument('--train-log', required=True, help="Train setine ait ham olay log dosyası.")
    parser.add_argument('--train-output', required=True, help="Oluşturulacak filtrelenmiş train dosyası.")
    parser.add_argument('--test-log', required=True, help="Test setine ait ham olay log dosyası.")
    parser.add_argument('--test-output', required=True, help="Oluşturulacak filtrelenmiş test dosyası.")
    
    args = parser.parse_args()

    session_map = load_session_map(args.map_file)
    if not session_map:
        print("HATA: Oturum haritası yüklenemediği için işlem durduruldu.")
        return

    if args.train_log:
        convert_to_yandex_format_30_only(args.train_log, session_map, args.train_output)

    if args.test_log:
        convert_to_yandex_format_30_only(args.test_log, session_map, args.test_output)

if __name__ == "__main__":
    main()