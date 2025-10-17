import os
from tqdm import tqdm
from collections import defaultdict

def create_hard_test_set(train_path, test_path, session_map_path, output_path):
    """
    Train setinde tıklanmış (sorgu, ilan) çiftlerini bulur.
    Test setinde bu çiftlerden herhangi birini içeren OTURUMLARI tespit eder.
    Bu "sızdıran" oturumları hariç tutarak yeni ve "zor" bir test seti oluşturur.
    """
    print("="*60)
    print("'Zor' Test Seti Oluşturma Başlatıldı...")
    print("="*60)

    # --- Önceki script'lerden alınan yardımcı fonksiyonlar ---
    def get_clicked_pairs_from_raw_file(file_path, desc):
        events_by_search_id = defaultdict(lambda: {'query_info': None, 'clicks': set()})
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    event_type = parts[2]
                    search_id = parts[0]
                    if event_type == 'Q': events_by_search_id[search_id]['query_info'] = parts
                    elif event_type == 'C': events_by_search_id[search_id]['clicks'].add(parts[3])
                except IndexError: continue
        
        clicked_pairs = set()
        for search_id, data in tqdm(events_by_search_id.items(), desc=desc):
            query_info, clicks = data.get('query_info'), data.get('clicks')
            if query_info and clicks:
                query_id, doc_ids = query_info[3], query_info[5:]
                for doc_id in doc_ids:
                    if doc_id in clicks:
                        clicked_pairs.add((query_id, doc_id))
        return clicked_pairs

    def load_session_map(map_path):
        search_to_session = {}
        with open(map_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2 and parts[0].isdigit():
                    search_to_session[parts[0]] = parts[1]
        return search_to_session
    # --- Yardımcı fonksiyonlar sonu ---

    # 1. Train setindeki tüm tıklanmış çiftleri bul (Cevap anahtarımız)
    print("\n[Adım 1] Train setindeki cevap anahtarı okunuyor...")
    train_clicked_pairs = get_clicked_pairs_from_raw_file(train_path, "--> Train verisi taranıyor")
    print(f"--> Train setinde {len(train_clicked_pairs):,} adet tıklanmış çift bulundu.")

    # 2. Test setindeki hangi OTURUMLARIN sızıntılı olduğunu bul
    print("\n[Adım 2] Test setindeki sızıntılı oturumlar tespit ediliyor...")
    search_to_session = load_session_map(session_map_path)
    test_events_by_session = defaultdict(list)
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                search_id = line.strip().split()[0]
                if search_id in search_to_session:
                    session_id = search_to_session[search_id]
                    test_events_by_session[session_id].append(line)
            except IndexError: continue

    contaminated_sessions = set()
    for session_id, lines in tqdm(test_events_by_session.items(), desc="--> Test oturumları kontrol ediliyor"):
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5 and parts[2] == 'Q':
                query_id = parts[3]
                clicked_docs_in_line = {p[3] for p in (l.strip().split() for l in lines) if len(p) > 3 and p[2] == 'C' and p[0] == parts[0]}
                for doc_id in parts[5:]:
                    if doc_id in clicked_docs_in_line and (query_id, doc_id) in train_clicked_pairs:
                        contaminated_sessions.add(session_id)
                        break # Bu oturum sızıntılı, diğer satırlarına bakmaya gerek yok
            if session_id in contaminated_sessions:
                break
    
    print(f"--> Test setindeki {len(test_events_by_session):,} oturumdan {len(contaminated_sessions):,} tanesinin sızıntılı olduğu bulundu.")

    # 3. Sızıntılı olmayan oturumları yeni dosyaya yaz
    print(f"\n[Adım 3] Temizlenmiş test seti '{output_path}' dosyasına yazılıyor...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for session_id, lines in tqdm(test_events_by_session.items(), desc="--> Temiz oturumlar yazılıyor"):
            if session_id not in contaminated_sessions:
                for line in lines:
                    f_out.write(line)
    
    print("\n✅ İşlem tamamlandı!")

if __name__ == '__main__':
    create_hard_test_set(
        train_path='raw_data/train.txt',
        test_path='raw_data/test.txt',
        session_map_path='raw_data/session_map.csv',
        output_path='raw_data/test_hard.txt'
    )