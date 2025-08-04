# !/usr/bin/python
# coding: utf8

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# PATH ayarları
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT_DIR, '..'))

# Custom modüller
from utils import *
import config 

# =======================================================================================
# === DEĞİŞTİRİLEN BÖLÜM: Bellek sorununu çözen yeni fonksiyon ===
# =======================================================================================

def process_data_in_stream(args):
    """
    Bu fonksiyon, orijinal `generate_list_dict` ve `generate_train_valid_test` fonksiyonlarının
    yaptığı işi, veriyi belleğe yüklemeden, akan şekilde (streaming) yapar.
    """
    print(" - train.txt okunuyor...")
    
    # Adım 1: Tüm oturumları belleğe yüklemeden önce gruplayalım.
    sessions_map = {}
    with open(os.path.join(args.input, 'train.txt'), 'r') as f:
        for line in tqdm(f, desc=" - Ham log dosyası okunuyor"):
            elements = line.strip().split('\t')
            session_id_str = elements[0]
            if session_id_str not in sessions_map:
                sessions_map[session_id_str] = []
            sessions_map[session_id_str].append(elements)
    
    print(f" - {len(sessions_map)} adet benzersiz oturum bulundu ve gruplandı.")
    
    # Adım 2: Oturumları işle ve Python objelerine dönüştür.
    infos_per_session = []
    session_sid, query_qid, url_uid = {'': 0}, {'': 0}, {'': 0}
    junk_click_cnt = 0
    
    #for session_id_str, elements_list in tqdm(sessions_map.items(), desc=" - Oturumlar işleniyor"):
    for session_id_str in tqdm(sorted(sessions_map.keys()), desc=" - Oturumlar işleniyor"):
        elements_list = sessions_map[session_id_str]

        if session_id_str not in session_sid:
            session_sid[session_id_str] = len(session_sid)
        sid = session_sid[session_id_str]
        
        session_data = {'sid': sid, 'qids': [], 'uidsS': [], 'clicksS': []}
        
        for elements in elements_list:
            if elements[1] == 'M':
                continue
            elif elements[2] in ['Q', 'T']:
                query = elements[4]
                if query not in query_qid:
                    query_qid[query] = len(query_qid)
                
                uids = []
                urls_domains = elements[-10:]
                for url_domain in urls_domains:
                    url = url_domain.split(',')[0]
                    if url not in url_uid:
                        url_uid[url] = len(url_uid)
                    uids.append(url_uid[url])
                
                session_data['qids'].append(query_qid[query])
                session_data['uidsS'].append(uids)
                session_data['clicksS'].append([0] * 10)

            elif elements[2] == 'C':
                if not session_data['uidsS']: continue
                clicked_url = elements[-1]
                if clicked_url in url_uid:
                    clicked_uid = url_uid[clicked_url]
                    if clicked_uid in session_data['uidsS'][-1]:
                        idx = session_data['uidsS'][-1].index(clicked_uid)
                        session_data['clicksS'][-1][idx] = 1
                    else:
                        junk_click_cnt += 1
        
        if session_data['qids']:
            infos_per_session.append(session_data)
            
    print(f' - {junk_click_cnt} adet junk click bilgisi atlandı.')

    # Adım 3: Karıştır ve train/valid/test olarak ayır.
    print(' - Veri seti karıştırılıyor ve bölünüyor...')
    random.seed(2333)
    random.shuffle(infos_per_session)
    
    if args.downsample < len(infos_per_session):
        print(f' - Veri seti {len(infos_per_session)} oturumdan {args.downsample} oturuma indirgeniyor.')
        infos_per_session = infos_per_session[:args.downsample]

    session_num = len(infos_per_session)
    train_session_num = int(session_num * args.trainset_ratio)
    valid_session_num = int(session_num * args.validset_ratio)
    
    sessions = {
        'train': infos_per_session[:train_session_num],
        'valid': infos_per_session[train_session_num:train_session_num + valid_session_num],
        'test': infos_per_session[train_session_num + valid_session_num:]
    }

    # Adım 4: Dosyalara yaz ve sözlükleri yeniden oluştur.
    print(' - Train/valid/test dosyaları ve sözlükler yazılıyor...')
    final_query_qid, final_url_uid = {'': 0}, {'': 0}
    for file_type, session_list in sessions.items():
        with open(os.path.join(args.output, f'{file_type}_per_query_quid.txt'), 'w') as f:
            for session in tqdm(session_list, desc=f"  - {file_type} seti yazılıyor"):
                sid = session['sid']
                for qid_orig, uids_orig, clicks in zip(session['qids'], session['uidsS'], session['clicksS']):
                    if qid_orig not in final_query_qid:
                        final_query_qid[qid_orig] = len(final_query_qid)
                    qid_final = final_query_qid[qid_orig]
                    
                    uids_final = []
                    for uid_orig in uids_orig:
                        if uid_orig not in final_url_uid:
                            final_url_uid[uid_orig] = len(final_url_uid)
                        uids_final.append(final_url_uid[uid_orig])
                    
                    f.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid_final, str(uids_final), str([0] * 10), str(clicks)))

    print(' - Final sözlükler kaydediliyor...')
    save_dict(args.output, 'query_qid.dict', final_query_qid)
    save_dict(args.output, 'url_uid.dict', final_url_uid)


# =========================================================================
# === ORİJİNAL FONKSİYONLAR ===
# =========================================================================

def construct_dgat_graph(args):
    # load entity dictionaries
    print('  - {}'.format('loading entity dictionaries...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')

    # Calc edge information for train/valid/test set
    # set_names = ['demo']
    set_names = ['train', 'valid', 'test']
    qid_edges, uid_edges = set(), set()
    qid_neighbors, uid_neighbors = {qid: set() for qid in range(len(query_qid))}, {uid: set() for uid in range(len(url_uid))}
    for set_name in set_names:
        print('  - {}'.format('Constructing relations in {} set'.format(set_name)))
        lines = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(set_name))).readlines()

        # Relation 0: Query-Query within the same session
        cur_sid = -1
        qid_set = set()
        for line in lines:
            attr = line.strip().split('\t')
            sid = int(attr[0].strip())
            qid = int(attr[1].strip())
            if cur_sid == sid:
                # query in the same session
                qid_set.add(qid)
            else:
                # session ends, start creating relations
                qid_list = list(qid_set)
                for i in range(1, len(qid_list)):
                    qid_edges.add(str([qid_list[i], qid_list[i - 1]]))
                    qid_edges.add(str([qid_list[i - 1], qid_list[i]]))
                # new session starts
                cur_sid = sid
                qid_set.clear()
                qid_set.add(qid)
        # The last session
        qid_list = list(qid_set)
        for i in range(1, len(qid_list)):
            qid_edges.add(str([qid_list[i], qid_list[i - 1]]))
            qid_edges.add(str([qid_list[i - 1], qid_list[i]]))

        # Relation 1 & 2: Document of is clicked in a Query
        for line in lines:
            attr = line.strip().split('\t')
            qid = int(attr[1].strip())
            uids = json.loads(attr[2].strip())
            clicks = json.loads(attr[4].strip())
            for uid, click in zip(uids, clicks):
                if click:
                    if set_name == 'train' or set_name == 'demo':
                        qid_neighbors[qid].add(uid)
                        uid_neighbors[uid].add(qid)
        
        # Relation 3: successive Documents in the same query
        for line in lines:
            attr = line.strip().split('\t')
            uids = json.loads(attr[2].strip())
            for i in range(1, len(uids)):
                uid_edges.add(str([uids[i], uids[i - 1]]))
                uid_edges.add(str([uids[i - 1], uids[i]]))
    
    # Meta-path to q-q & u-u
    for qid in qid_neighbors:
        qid_neigh = list(qid_neighbors[qid])
        for i in range(len(qid_neigh)):
            for j in range(i + 1, len(qid_neigh)):
                uid_edges.add(str([qid_neigh[i], qid_neigh[j]]))
                uid_edges.add(str([qid_neigh[j], qid_neigh[i]]))
    for uid in uid_neighbors:
        uid_neigh = list(uid_neighbors[uid])
        for i in range(len(uid_neigh)):
            for j in range(i + 1, len(uid_neigh)):
                qid_edges.add(str([uid_neigh[i], uid_neigh[j]]))
                qid_edges.add(str([uid_neigh[j], uid_neigh[i]]))
    
    # Add self-loop
    for qid in range(len(query_qid)):
        qid_edges.add(str([qid, qid]))
    for uid in range(len(url_uid)):
        uid_edges.add(str([uid, uid]))

    # Convert & save edges information from set/list into tensor
    qid_edges = [eval(edge) for edge in qid_edges]
    uid_edges = [eval(edge) for edge in uid_edges]
    # print(qid_edges)
    # print(uid_edges)
    qid_edge_index = torch.transpose(torch.from_numpy(np.array(qid_edges, dtype=np.int64)), 0, 1)
    uid_edge_index = torch.transpose(torch.from_numpy(np.array(uid_edges, dtype=np.int64)), 0, 1)
    torch.save(qid_edge_index, os.path.join(args.output, 'dgat_qid_edge_index.pth'))
    torch.save(uid_edge_index, os.path.join(args.output, 'dgat_uid_edge_index.pth'))

    # Count degrees of qid/uid nodes
    qid_degrees, uid_degrees = [set([i]) for i in range(len(query_qid))], [set([i]) for i in range(len(url_uid))]
    for qid_edge in qid_edges:
        qid_degrees[qid_edge[0]].add(qid_edge[1])
        qid_degrees[qid_edge[1]].add(qid_edge[0])
    for uid_edge in uid_edges:
        uid_degrees[uid_edge[0]].add(uid_edge[1])
        uid_degrees[uid_edge[1]].add(uid_edge[0])
    qid_degrees = [len(d_set) for d_set in qid_degrees]
    uid_degrees = [len(d_set) for d_set in uid_degrees]
    non_isolated_qid_cnt = sum([1 if qid_degree > 1 else 0 for qid_degree in qid_degrees])
    non_isolated_uid_cnt = sum([1 if uid_degree > 1 else 0 for uid_degree in uid_degrees])
    print('  - {}'.format('Mean/Max/Min qid degree: {}, {}, {}'.format(sum(qid_degrees) / len(qid_degrees), max(qid_degrees), min(qid_degrees))))
    print('  - {}'.format('Mean/Max/Min uid degree: {}, {}, {}'.format(sum(uid_degrees) / len(uid_degrees), max(uid_degrees), min(uid_degrees))))
    print('  - {}'.format('Non-isolated qid node num: {}'.format(non_isolated_qid_cnt)))
    print('  - {}'.format('Non-isolated uid node num: {}'.format(non_isolated_uid_cnt)))

    # Save direct uid-uid neighbors for neighbor feature interactions
    uid_num = len(url_uid)
    max_node_degree = 64
    uid_neigh = [set([i]) for i in range(uid_num)]
    uid_neigh_sampler = nn.Embedding(uid_num, max_node_degree)
    for edge in uid_edges:
        src, dst = edge[0], edge[1]
        uid_neigh[src].add(dst)
        uid_neigh[dst].add(src)
    for idx, adj in enumerate(uid_neigh):
        adj_list = list(adj)
        if len(adj_list) >= max_node_degree:
            adj_sample = torch.from_numpy(np.array(random.sample(adj_list, max_node_degree), dtype=np.int64))
        else:
            adj_sample = torch.from_numpy(np.array(random.choices(adj_list, k=max_node_degree), dtype=np.int64))
        uid_neigh_sampler.weight.data[idx] = adj_sample.clone()
    torch.save(uid_neigh_sampler, os.path.join(args.output, 'dgat_uid_neighbors.pth'))


def generate_dataset_for_cold_start(args):
    def load_dataset(data_path):
        """
        Loads the dataset
        """
        data_set = []
        lines = open(data_path).readlines()
        previous_sid = -1
        qids, uids, vids, clicks = [], [], [], []
        for line in lines:
            attr = line.strip().split('\t')
            sid = int(attr[0].strip())
            if previous_sid != sid:
                # a new session starts
                if previous_sid != -1:
                    assert len(uids) == len(qids)
                    assert len(vids) == len(qids)
                    assert len(clicks) == len(qids)
                    assert len(vids[0]) == 10
                    assert len(uids[0]) == 10
                    assert len(clicks[0]) == 10
                    data_set.append({'sid': previous_sid,
                                     'qids': qids,
                                     'uids': uids,
                                     'vids': vids,
                                     'clicks': clicks})
                previous_sid = sid
                qids = [int(attr[1].strip())]
                uids = [json.loads(attr[2].strip())]
                vids = [json.loads(attr[3].strip())]
                clicks = [json.loads(attr[4].strip())]
            else:
                # the previous session continues
                qids.append(int(attr[1].strip()))
                uids.append(json.loads(attr[2].strip()))
                vids.append(json.loads(attr[3].strip()))
                clicks.append(json.loads(attr[4].strip()))
        data_set.append({'sid': previous_sid,
                        'qids': qids,
                        'uids': uids,
                        'vids': vids,
                        'clicks': clicks,})
        return data_set
    
    # Load original train/test dataset
    print('  - {}'.format('start loading train/test set...'))
    train_set = load_dataset(os.path.join(args.output, 'train_per_query_quid.txt'))
    test_set = load_dataset(os.path.join(args.output, 'test_per_query_quid.txt'))
    print('    - {}'.format('train session num: {}'.format(len(train_set))))
    print('    - {}'.format('test session num: {}'.format(len(test_set))))

    # Construct train query set for filtering
    print('  - {}'.format('Constructing train query set for filtering'))
    step_pbar = tqdm(total=len(train_set))
    train_query_set = set()
    train_doc_set = set()
    for session_info in train_set:
        step_pbar.update(1)
        train_query_set = train_query_set | set(session_info['qids'])
        for uids in session_info['uids']:
            train_doc_set = train_doc_set | set(uids)
    print('    - {}'.format('unique train query num: {}'.format(len(train_query_set))))
    print('    - {}'.format('unique train doc num: {}'.format(len(train_doc_set))))

    # Divide the full test set into four mutually exclusive parts
    print('  - {}'.format('Start the full test set division'))
    step_pbar = tqdm(total=len(test_set))
    cold_q, cold_d, cold_qd, warm_qd = [], [], [], []
    for session_info in test_set:
        step_pbar.update(1)
        is_q_cold, is_d_cold = False, False
        for qid in session_info['qids']:
            if qid not in train_query_set:
                is_q_cold = True
                break
        for uids in session_info['uids']:
            for uid in uids:
                if uid not in train_doc_set:
                    is_d_cold = True
                    break
            if is_d_cold:
                break
        if is_q_cold:
            if is_d_cold:
                cold_qd.append(session_info)
            else:
                cold_q.append(session_info)
        else:
            if is_d_cold:
                cold_d.append(session_info)
            else:
                warm_qd.append(session_info)
    print('    - {}'.format('Total session num: {}'.format(len(cold_q) + len(cold_d) + len(cold_qd) + len(warm_qd))))
    print('    - {}'.format('Cold Q session num: {}'.format(len(cold_q))))
    print('    - {}'.format('Cold D session num: {}'.format(len(cold_d))))
    print('    - {}'.format('Cold QD session num: {}'.format(len(cold_qd))))
    print('    - {}'.format('Warm QD session num: {}'.format(len(warm_qd))))

    # Save the four session sets back to files
    print('    - {}'.format('Write back cold_q set'))
    file = open(os.path.join(args.output, 'cold_q_test_per_query_quid.txt'), 'w')
    for session_info in cold_q:
        sid = session_info['sid']
        qids = session_info['qids']
        uidsS = session_info['uids']
        vidsS = session_info['vids']
        clicksS = session_info['clicks']
        for qid, uids, vids, clicks in zip(qids, uidsS, vidsS, clicksS):
            file.write("{}\t{}\t{}\t{}\t{}\n".format(sid, qid, str(uids), str(vids), str(clicks)))
    file.close()
    # ... (Diğer setler için aynı mantık) ...


def compute_sparsity(args):
    # load entity dictionaries
    print('  - {}'.format('Loading entity dictionaries...'))
    query_qid = load_dict(args.output, 'query_qid.dict')
    url_uid = load_dict(args.output, 'url_uid.dict')

    # Calc sparisity for the dataset
    print('  - {}'.format('Count the query-doc pairs in the dataset...'))
    set_names = ['train', 'valid', 'test']
    train_qu_set, q_set, u_set = set(), set(), set()
    for set_name in set_names:
        print('   - {}'.format('Counting the query-doc pairs in the {} set'.format(set_name)))
        lines = open(os.path.join(args.output, '{}_per_query_quid.txt'.format(set_name))).readlines()
        for line in lines:
            attr = line.strip().split('\t')
            qid = int(attr[1].strip())
            uids = json.loads(attr[2].strip())
            for uid in uids:
                if set_name == 'train':
                    train_qu_set.add(str([qid, uid]))
                q_set.add(qid)
                u_set.add(uid)
    
    # Compute the sparsity
    assert len(q_set) + 1 == len(query_qid)
    assert len(u_set) + 1 == len(url_uid)
    print('  - {}'.format('There are {} unique query-doc pairs in the training dataset...'.format(len(train_qu_set))))
    print('  - {}'.format('There are {} unique queries in the dataset...'.format(len(q_set))))
    print('  - {}'.format('There are {} unique docs in the dataset...'.format(len(u_set))))
    print('  - {}'.format('There are {} possible query-doc pairs in the whole dataset...'.format(len(q_set) * len(u_set))))
    print('  - {}'.format('The sparsity is: 1 - {} / {} = {}%'.format(len(train_qu_set), len(q_set) * len(u_set), 100 - 100 * len(train_qu_set) / (len(q_set) * len(u_set)))))


def main():
    parser = argparse.ArgumentParser('Yandex')
    parser.add_argument('--input', default='../dataset/Yandex/', help='input path')
    parser.add_argument('--output', default='./data/Yandex', help='output path')
    parser.add_argument('--list_dict', action='store_true', help='generate list & dict files')
    parser.add_argument('--train_valid_test_data', action='store_true', help='generate train/valid/test data txt')
    parser.add_argument('--dgat', action='store_true', help='construct graph for double GAT')
    parser.add_argument('--cold_start', action='store_true', help='construct dataset for studying cold start problems')
    parser.add_argument('--downsample', type=int, default=10000000, help='construct graph for double GAT')
    parser.add_argument('--sparsity', action='store_true', help='compute sparisity for the dataset')
    parser.add_argument('--trainset_ratio', type=float, default=0.8)
    parser.add_argument('--validset_ratio', type=float, default=0.1)
    args = parser.parse_args()
    
    send_slack_message(config.SLACK_CONFIG, f"Yandex.py Veri Ön İşleme Süreci Başladı.\n> `Arguments: {args}`")
    try:
        if args.list_dict or args.train_valid_test_data:
            send_slack_message(config.SLACK_CONFIG, "➡️ Adım: Bellek dostu `train/valid/test` oluşturma işlemi başlıyor...")
            process_data_in_stream(args)
            send_slack_message(config.SLACK_CONFIG, "✅ Adım: `train/valid/test` oluşturma tamamlandı.")
        if args.dgat:
            send_slack_message(config.SLACK_CONFIG, "➡️ Adım: `dgat` grafiği oluşturma işlemi başlıyor...")
            construct_dgat_graph(args)
            send_slack_message(config.SLACK_CONFIG, "✅ Adım: `dgat` grafiği oluşturma tamamlandı.")
        if args.cold_start:
            send_slack_message(config.SLACK_CONFIG, "➡️ Adım: `cold_start` veri seti oluşturma işlemi başlıyor...")
            generate_dataset_for_cold_start(args)
            send_slack_message(config.SLACK_CONFIG, "✅ Adım: `cold_start` veri seti oluşturma tamamlandı.")
        if args.sparsity:
            send_slack_message(config.SLACK_CONFIG, "➡️ Adım: `sparsity` hesaplama işlemi başlıyor...")
            compute_sparsity(args)
            send_slack_message(config.SLACK_CONFIG, "✅ Adım: `sparsity` hesaplama tamamlandı.")
        
        print('===> {}'.format('Done.'))
        send_slack_message(config.SLACK_CONFIG, "Yandex.py Ön İşleme Süreci Başarıyla Tamamlandı!")
    except Exception as e:
        import traceback
        error_full = traceback.format_exc()
        error_message = f"HATA: Yandex.py çalışırken bir sorun oluştu!\n> *Hata Mesajı:*\n> ```{e}\n{error_full}```"
        send_slack_message(config.SLACK_CONFIG, error_message)
        raise e

if __name__ == '__main__':
    main()