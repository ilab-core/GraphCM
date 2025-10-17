# !/usr/bin/python
# coding: utf8
import os, sys, json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))

# --- basit dict yardımcıları 
def save_dict(out_dir, name, obj):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, name), 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False)

def load_dict(out_dir, name):
    with open(os.path.join(out_dir, name), 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------------
# 1) Union sözlükleri (train+test)
# ---------------------------
def _scan_for_dicts(fp, query_qid, url_uid):
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            el = line.strip().split('\t')
            if len(el) < 3:
                continue
            if el[2] in 'Q':
                q = el[4]
                if q and q != '0' and q not in query_qid:
                    query_qid[q] = len(query_qid)
                # URL tarafında padding görürsek sözlüğe eklemiyoruz
                for ud in el[-30:]:
                    url = ud.split(',')[0]
                    if not url or url == '0' or url.upper() == 'PAD':
                        continue
                    if url not in url_uid:
                        url_uid[url] = len(url_uid)

def build_union_dicts(train_fp, test_fp, out_dir):
    print(" - ID sözlükleri union(train+test) üzerinden çıkarılıyor...")
    query_qid, url_uid = {'': 0}, {'': 0}
    _scan_for_dicts(train_fp, query_qid, url_uid)
    _scan_for_dicts(test_fp,  query_qid, url_uid)
    save_dict(out_dir, 'query_qid.dict', query_qid)
    save_dict(out_dir, 'url_uid.dict',   url_uid)
    print(f" - Dicts saved → queries={len(query_qid)}, urls={len(url_uid)}")

# -----------------------------------------
# 2) Train/Test txt → *_per_query_quid.txt
# -----------------------------------------
def _group_sessions(fp):
    m = {}
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            el = line.strip().split('\t')
            if not el: continue
            sid = el[0]
            m.setdefault(sid, []).append(el)
    return m

def _is_pad_url(url: str) -> bool:
    return (not url) or (url == '0') or (url.upper() == 'PAD')

def write_split(input_fp, out_fp, query_qid, url_uid, tag='train'):
    print(f" - {tag}.txt okunuyor ve {os.path.basename(out_fp)} yazılıyor...")
    sessions = _group_sessions(input_fp)
    with open(out_fp, 'w', encoding='utf-8') as out:
        junk_click_cnt = 0
        for sid_str in tqdm(sorted(sessions.keys()), desc=f"  - {tag} oturumları"):
            seq = sessions[sid_str]
            sid_num = int(sid_str) if sid_str.isdigit() else (abs(hash(f"{tag}:{sid_str}")) % (10**9) + (0 if tag=='train' else 10**9))
            qids, uidsS, clicksS = [], [], []
            for el in seq:
                if len(el) < 3: 
                    continue
                if el[1] == 'M':
                    continue
                elif el[2] == 'Q':
                    q = el[4]
                    if q not in query_qid:
                        continue
                    qid = query_qid[q]

                    # --- URL → UID map (padding görünce direkt 0 atıyoruz) ---
                    uids = []
                    for ud in el[-30:]:
                        url = ud.split(',')[0]
                        if _is_pad_url(url):
                            uids.append(0)                 
                        else:
                            if url not in url_uid:
                                uids = None
                                break
                            uids.append(url_uid[url])
                    if uids is None:
                        continue

                    qids.append(qid)
                    clicksS.append([0]*30)
                    uidsS.append(uids)

                elif el[2] == 'C':
                    if not uidsS:
                        continue
                    clicked_url = el[-1]
                    # PAD tıklama varsa yok say (UID=0 zaten padding)
                    if _is_pad_url(clicked_url):
                        continue
                    if clicked_url in url_uid:
                        uid = url_uid[clicked_url]
                        if uid in uidsS[-1]:
                            idx = uidsS[-1].index(uid)
                            clicksS[-1][idx] = 1
                        else:
                            junk_click_cnt += 1

            for qid, uids, clicks in zip(qids, uidsS, clicksS):
                out.write(f"{sid_num}\t{qid}\t{str(uids)}\t{str([1]*30)}\t{str(clicks)}\n")
    print(f" - {os.path.basename(out_fp)} yazıldı.")

# -----------------------------------------
# 3) Graf (transdüktif): train+test yapısal
# -----------------------------------------
def construct_dgat_graph(out_dir):
    print('  - Sözlükler yükleniyor...')
    query_qid = load_dict(out_dir, 'query_qid.dict')
    url_uid   = load_dict(out_dir, 'url_uid.dict')

    set_names = []
    for s in ['train', 'valid', 'test']:  # valid şu anlık boş yaratılıyor.
        fp = os.path.join(out_dir, f'{s}_per_query_quid.txt')
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            set_names.append(s)
    print(f'  - Graf setleri: {set_names}')

    qid_edges, uid_edges = set(), set()
    qid_neighbors = {qid: set() for qid in range(len(query_qid))}
    uid_neighbors = {uid: set() for uid in range(len(url_uid))}

    for set_name in set_names:
        lines = open(os.path.join(out_dir, f'{set_name}_per_query_quid.txt'), 'r', encoding='utf-8').read().splitlines()

        # Relation 0: Query-Query (same session) — train+test(+valid eğer boş değilse)
        cur_sid, cur_qs = None, []
        for ln in lines:
            attr = ln.split('\t')
            sid = int(attr[0]); qid = int(attr[1])
            if cur_sid is None or cur_sid == sid:
                cur_sid = sid; cur_qs.append(qid)
            else:
                for i in range(1, len(cur_qs)):
                    qid_edges.add(str([cur_qs[i],   cur_qs[i-1]]))
                    qid_edges.add(str([cur_qs[i-1], cur_qs[i]]))
                cur_sid, cur_qs = sid, [qid]
        for i in range(1, len(cur_qs)):
            qid_edges.add(str([cur_qs[i],   cur_qs[i-1]]))
            qid_edges.add(str([cur_qs[i-1], cur_qs[i]]))

        # Relation 1&2: clicked Q–U — sadece train
        if set_name == 'train':
            for ln in lines:
                _, qid, uids, _, clicks = ln.split('\t')
                qid   = int(qid)
                uids  = json.loads(uids)
                clicks= json.loads(clicks)
                for uid, c in zip(uids, clicks):
                    if c:
                        qid_neighbors[qid].add(uid)
                        uid_neighbors[uid].add(qid)

        # Relation 3: successive docs in a query — train+test(+valid eğer boş değilse)
        for ln in lines:
            _, _, uids, _, _ = ln.split('\t')
            uids = json.loads(uids)
            for i in range(1, len(uids)):
                uid_edges.add(str([uids[i],   uids[i-1]]))
                uid_edges.add(str([uids[i-1], uids[i]]))

    # Meta-path: q-q & u-u (yalnız train’den öğrenilen Q–U komşuluklarına göre)
    for q in qid_neighbors:
        neigh = list(qid_neighbors[q])
        for i in range(len(neigh)):
            for j in range(i+1, len(neigh)):
                uid_edges.add(str([neigh[i], neigh[j]]))
                uid_edges.add(str([neigh[j], neigh[i]]))
    for u in uid_neighbors:
        neigh = list(uid_neighbors[u])
        for i in range(len(neigh)):
            for j in range(i+1, len(neigh)):
                qid_edges.add(str([neigh[i], neigh[j]]))
                qid_edges.add(str([neigh[j], neigh[i]]))

    # Self-loop
    for q in range(len(query_qid)):
        qid_edges.add(str([q, q]))
    for u in range(len(url_uid)):
        uid_edges.add(str([u, u]))

    # Tensöre çevir ve kaydet
    qid_edges = [eval(e) for e in qid_edges]
    uid_edges = [eval(e) for e in uid_edges]
    qid_edge_index = torch.transpose(torch.from_numpy(np.array(qid_edges, dtype=np.int64)), 0, 1)
    uid_edge_index = torch.transpose(torch.from_numpy(np.array(uid_edges, dtype=np.int64)), 0, 1)
    torch.save(qid_edge_index, os.path.join(out_dir, 'dgat_qid_edge_index.pth'))
    torch.save(uid_edge_index, os.path.join(out_dir, 'dgat_uid_edge_index.pth'))
    print('  - Graf kenar dosyaları kaydedildi.')

    # UID komşu örnekleyici 
    uid_num = len(url_uid); max_node_degree = 64
    uid_neigh = [set([i]) for i in range(uid_num)]
    uid_neigh_sampler = nn.Embedding(uid_num, max_node_degree)
    for e in uid_edges:
        src, dst = e[0], e[1]
        uid_neigh[src].add(dst); uid_neigh[dst].add(src)
    for idx, adj in enumerate(uid_neigh):
        adj_list = list(adj)
        if len(adj_list) >= max_node_degree:
            sample = np.array(random.sample(adj_list, max_node_degree), dtype=np.int64)
        else:
            sample = np.array(random.choices(adj_list, k=max_node_degree), dtype=np.int64)
        uid_neigh_sampler.weight.data[idx] = torch.from_numpy(sample).clone()
    torch.save(uid_neigh_sampler, os.path.join(out_dir, 'dgat_uid_neighbors.pth'))
    print('  - UID komşu örnekleyici kaydedildi.')


def main():
    parser = argparse.ArgumentParser('final_preprocess (two-input, transductive, pad=0)')
    parser.add_argument('--train_fp', default='raw_data/train_final.txt')
    parser.add_argument('--test_fp',  default='raw_data/test_final.txt')
    parser.add_argument('--output',   default='data/emj_train_test')
    parser.add_argument('--build_graph', action='store_true', help='DGAT grafını kur')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 1) Sözlükleri çıkar (union)
    build_union_dicts(args.train_fp, args.test_fp, args.output)

    # 2) Train/Test çıktılarını yaz (padding görünce UID=0 ata)
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid   = load_dict(args.output, 'url_uid.dict')
    write_split(args.train_fp, os.path.join(args.output, 'train_per_query_quid.txt'), query_qid, url_uid, tag='train')
    write_split(args.test_fp,  os.path.join(args.output, 'test_per_query_quid.txt'),  query_qid, url_uid, tag='test')

    # 2.5) boş valid dosyası oluştur
    open(os.path.join(args.output, 'valid_per_query_quid.txt'), 'w', encoding='utf-8').close()

    # 2.6) vtype_vid.dict dosyasını oluştur
    print(" - vtype_vid.dict dosyası oluşturuluyor...")
    save_dict(args.output, 'vtype_vid.dict', {'': 0, '0': 1}) # elimizde vtype olmadığından bu şekilde oluşturuldu.

    # 3) Graf 
    if args.build_graph:
        construct_dgat_graph(args.output)

    print('===> Preprocess tamam.')

if __name__ == '__main__':
    main()
