# -*- coding: utf-8 -*-
import collections
import os
import re

# --- AYARLAR ---
INPUT_FILE_NAME = 'data/MyDataset/raw_log.txt'
OUTPUT_FILE_NAME = 'data/MyDataset/train.txt'
# --- AYARLAR BİTTİ ---

def process_logs_ultimate_fix():
    """
    Bu betik, standart split() fonksiyonunu tamamen terk eder.
    Her satırı ve sütunu, görünmez karakterlere karşı dayanıklı olan
    Regex ile parçalar ve işler. Bu, sorunu kesin olarak çözer.
    """
    Q_PATTERN = re.compile(r'^(\S+)\s+(\S+)\s+(Q)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)$')
    DOC_PATTERN = re.compile(r'(\d+,\d+)')
    M_PATTERN = re.compile(r'^(\S+)\s+(M)\s+(\S+)\s+(\S+)$')
    C_PATTERN = re.compile(r'^(\S+)\s+(\S+)\s+(C)\s+(\S+)\s+(\S+)$')

    output_dir = os.path.dirname(OUTPUT_FILE_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"'{INPUT_FILE_NAME}' okunuyor...")
    sessions = collections.defaultdict(list)
    try:
        with open(INPUT_FILE_NAME, 'r', encoding='utf-8') as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    match = re.match(r'^(\S+)', stripped_line)
                    if match:
                        sessions[match.group(1)].append(stripped_line)
    except FileNotFoundError:
        print(f"HATA: '{INPUT_FILE_NAME}' bulunamadı.")
        return

    print(f"{len(sessions)} adet benzersiz oturum bulundu. Veri işleniyor...")

    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f_out:
        for session_id, lines in sessions.items():
            valid_queries = {}
            for line in lines:
                q_match = Q_PATTERN.match(line)
                if q_match:
                    doc_string = q_match.group(7)
                    ilan_listesi = DOC_PATTERN.findall(doc_string)

                    if len(ilan_listesi) >= 10:
                        serp_id = q_match.group(4)
                        top_10_ilan_ciftleri = ilan_listesi[:10]
                        top_10_doc_ids = {doc.split(',')[0] for doc in top_10_ilan_ciftleri}
                        valid_queries[serp_id] = (top_10_doc_ids, top_10_ilan_ciftleri)

            if not valid_queries:
                continue

            for line in lines:
                m_match = M_PATTERN.match(line)
                q_match = Q_PATTERN.match(line)
                c_match = C_PATTERN.match(line)

                if m_match:
                    f_out.write("\t".join(m_match.groups()) + '\n')

                elif q_match:
                    serp_id = q_match.group(4)
                    if serp_id in valid_queries:
                        static_parts = q_match.groups()[:6]
                        _, truncated_ilan_listesi = valid_queries[serp_id]

                        static_parts_str = "\t".join(static_parts)
                        ilan_listesi_str = " ".join(truncated_ilan_listesi)  # <== BURADA FİX VAR

                        f_out.write(f"{static_parts_str}\t{ilan_listesi_str}\n")

                elif c_match:
                    serp_id = c_match.group(4)
                    if serp_id in valid_queries:
                        valid_doc_ids, _ = valid_queries[serp_id]
                        clicked_doc_id = c_match.group(5)
                        if clicked_doc_id in valid_doc_ids:
                            f_out.write("\t".join(c_match.groups()) + '\n')

    print(f"İşlem tamamlandı! Sonuçlar '{OUTPUT_FILE_NAME}' dosyasına yazıldı.")

if __name__ == '__main__':
    process_logs_ultimate_fix()
