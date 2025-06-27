import os
import json
import argparse
import random
from tqdm import tqdm
from utils import save_list, load_list, save_dict

def parse_yandex_format(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    session_id = 0
    infos_per_session = []
    query_qid, url_uid = {'': 0}, {'': 0}

    i = 0
    while i < len(lines):
        line = lines[i].strip().split("\t")
        if line[2] != "Q":
            i += 1
            continue

        qid_raw = line[3]
        ad_ids = line[5:]
        qid = query_qid.setdefault(qid_raw, len(query_qid))
        uids = [url_uid.setdefault(ad, len(url_uid)) for ad in ad_ids]
        clicks = [0] * len(uids)

        # check C lines
        i += 1
        while i < len(lines) and lines[i].strip().split("\t")[2] == "C":
            cline = lines[i].strip().split("\t")
            clicked_ad = cline[4]
            if clicked_ad in ad_ids:
                idx = ad_ids.index(clicked_ad)
                clicks[idx] = 1
            i += 1

        infos_per_session.append({
            'sid': session_id,
            'qids': [qid],
            'uidsS': [uids],
            'clicksS': [clicks]
        })
        session_id += 1

    return infos_per_session, query_qid, url_uid

def write_quid_txt_files(sessions, path, prefix, query_qid, url_uid):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{prefix}_per_query_quid.txt"), "w", encoding="utf-8") as f:
        for sess in sessions:
            sid = sess['sid']
            for qid, uids, clicks in zip(sess['qids'], sess['uidsS'], sess['clicksS']):
                # Doküman sayısını 10'a sabitle
                if len(uids) < 10:
                    pad_len = 10 - len(uids)
                    uids += [0] * pad_len
                    clicks += [0] * pad_len
                elif len(uids) > 10:
                    uids = uids[:10]
                    clicks = clicks[:10]

                qid_out = query_qid.get(str(qid), 0)
                uids_out = [url_uid.get(str(uid), 0) for uid in uids]
                f.write(f"{sid}\t{qid_out}\t{json.dumps(uids_out)}\t{json.dumps([0]*10)}\t{json.dumps(clicks)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_txt', type=str, default='data/yandex_format_emj.txt')
    parser.add_argument('--output_dir', type=str, default='data/emj')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    args = parser.parse_args()

    print("Parsing input...")
    sessions, query_qid, url_uid = parse_yandex_format(args.input_txt)

    print("Shuffling and splitting...")
    random.seed(2333)
    random.shuffle(sessions)
    N = len(sessions)
    N_train = int(N * args.train_ratio)
    N_valid = int(N * args.valid_ratio)

    train_set = sessions[:N_train]
    valid_set = sessions[N_train:N_train + N_valid]
    test_set = sessions[N_train + N_valid:]

    print("Saving infos_per_session.list...")
    os.makedirs(args.output_dir, exist_ok=True)
    save_list(args.output_dir, 'infos_per_session.list', sessions)

    print("Writing *_per_query_quid.txt files...")
    write_quid_txt_files(train_set, args.output_dir, 'train', query_qid, url_uid)
    write_quid_txt_files(valid_set, args.output_dir, 'valid', query_qid, url_uid)
    write_quid_txt_files(test_set, args.output_dir, 'test', query_qid, url_uid)

    print("Saving dictionaries...")
    save_dict(args.output_dir, 'query_qid.dict', query_qid)
    save_dict(args.output_dir, 'url_uid.dict', url_uid)

    print("✅ Done!")

if __name__ == '__main__':
    main()
