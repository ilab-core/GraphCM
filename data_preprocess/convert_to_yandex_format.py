import os
import pandas as pd
from glob import glob

SESSION_FOLDER = "my_data/session_data"
QUERY_FOLDER = "my_data/emj_2024_10"
OUTPUT_PATH = "data/yandex_format_emj.txt"

def load_session_map():
    """
    session_id (cs_id) ile ds_search_id eşlemesini döner
    """
    mapping = []
    for file in sorted(glob(os.path.join(SESSION_FOLDER, "*.csv"))):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            cs_id = row["cs_id"].split(".")[0]
            ds_search_id = str(row["ds_search_id"])
            mapping.append((cs_id, ds_search_id))
    return mapping

def load_query_logs():
    """
    ds_search_id'ye karşılık gelen görüntülenen ve tıklanan ad_id'leri döner
    """
    query_map = dict()
    for file in sorted(glob(os.path.join(QUERY_FOLDER, "query_id-*.txt"))):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                ds_id = parts[0]
                event_type = parts[2]
                if event_type == "Q":
                    ad_ids = [ad for ad in parts[4:] if ad != "0"]
                    query_map[ds_id] = {"ad_ids": ad_ids, "clicks": []}
                elif event_type == "C":
                    clicked_ad = parts[4]
                    query_map.setdefault(ds_id, {"ad_ids": [], "clicks": []})
                    query_map[ds_id]["clicks"].append(clicked_ad)
    return query_map

def convert_to_yandex_format():
    print("Loading session mappings...")
    session_map = load_session_map()
    print("Loading query logs...")
    query_logs = load_query_logs()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print("Writing output to Yandex format...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, (cs_id, ds_id) in enumerate(session_map):
            if ds_id not in query_logs:
                continue
            q = query_logs[ds_id]
            ad_list = q["ad_ids"]
            clicks = q["clicks"]

            # Q line: session_id, 0, Q, 0, ad_id1, ad_id2, ...
            out.write(f"{i}\t0\tQ\t0\t" + "\t".join(ad_list) + "\n")

            # C lines: session_id, 0, C, 0, clicked_ad
            for clicked in clicks:
                out.write(f"{i}\t0\tC\t0\t{clicked}\n")

    print(f"✅ Done! Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    convert_to_yandex_format()
