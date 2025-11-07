# -*- coding: utf-8 -*-
"""
Senaryo 2 (Bağlamsal Olasılık) için tahminler üretir.
'conditional_click_prob' sütunu ile karşılaştırılacaktır.

Bizim `test_per_query_quid.txt` dosyamızı okur, (quid, uids, vids, true_clicks)
bilgilerini alır ve modeli *gerçek tıklama geçmişiyle* besler.
"""

import torch
import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model
from dataset import Dataset
import utils

# Model parametreleri (diğeriyle aynı)
MODEL_PARAMS = {
    'dataset': 'emj_train_test',
    'model_dir': './outputs/models/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15', 'optim': 'adam',
    'algo': 'GraphCM', 'learning_rate': 0.0003, 'batch_size': 1024, # Tahmin için batch_size
    'embed_size': 32, 'hidden_size': 64, 'max_d_num': 30, 'vtype_embed_size': 8,
    'click_embed_size': 4, 'pos_embed_size': 4, 'combine': 'mul', 'use_gnn': True,
    'gnn_neigh_sample': 15, 'gnn_att_heads': 2, 'weight_decay': 1e-05, 'dropout_rate': 0.4,
    'load_model': 21150, 'momentum': 0.99, 'gnn_dropout': 0, 'gnn_leaky_slope': 0.2, 
    'gnn_concat': False, 'inter_neigh_sample': 0, 'inter_leaky_slope': 0.2, 'gpu_num': 1,
    'data_parallel': False, 'num_steps': 28200, 'eval_freq': 1410, 'check_point': 1410,
    'patience': 5, 'lr_decay': 0.5, 'num_iter': 1, 'reg_relevance': 1.0, 
    'use_pretrain_embed': False, 'train': False, 'valid': False, 'test': False, 'rank': False,
    'result_dir': './outputs/results/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'summary_dir': './outputs/summary/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'log_dir': './outputs/log/',
}

# --- Girdi ve Çıktı Dosyaları ---
OUR_TEST_FILE = 'data/emj_train_test/test_per_query_quid.txt' # Girdi
OUTPUT_CSV_FILE = 'compare/GraphCM_S2_Contextual_Predictions.csv' # Çıktı

def load_model_and_device(args):
    """Modeli ve cihazı yükler."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Cihaz kullanılıyor: {device}")
    print("-> Veri seti sözlükleri ve model yapısı yükleniyor...")
    dataset = Dataset(args)
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    print(f"-> Model checkpoint'i yükleniyor (Adım: {args.load_model})...")
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    model.model.to(device)
    model.model.eval()
    return model.model, device

def parse_our_test_line(line, max_d_num):
    """test_per_query_quid.txt dosyasından bir satırı parse eder."""
    try:
        parts = line.strip().split('\t')
        if len(parts) != 5: return None
        quid = int(parts[1])
        uids = json.loads(parts[2])
        vids = json.loads(parts[3])
        clicks = json.loads(parts[4]) # [c_1, ..., c_30]
        
        if len(uids) == max_d_num and len(vids) == max_d_num and len(clicks) == max_d_num:
            return quid, uids, vids, clicks
        else:
            return None
    except Exception:
        return None

def main():
    print("GraphCM Senaryo 2 (Bağlamsal) Tahmin Script'i Başlatıldı.")
    
    model_args = argparse.Namespace(**MODEL_PARAMS)
    model, device = load_model_and_device(model_args)
    max_d_num = model_args.max_d_num
    
    results = [] # (quid, uid, rank, prob_s2)
    batch_size = model_args.batch_size
    
    try:
        with open(OUR_TEST_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"HATA: Bizim test dosyamız bulunamadı: {OUR_TEST_FILE}")
        return

    print(f"-> {len(lines)} sorgu (satır) bizim test setimizden işlenecek...")
    
    # Batch'ler halinde işlemek için
    all_parsed_lines = []
    for line in lines:
        parsed = parse_our_test_line(line, max_d_num)
        if parsed:
            all_parsed_lines.append(parsed)

    with torch.no_grad():
        for i in tqdm(range(0, len(all_parsed_lines), batch_size), desc="S2 Batch'leri işleniyor"):
            batch_lines = all_parsed_lines[i:i+batch_size]
            
            qids_s2 = []
            uids_s2 = []
            vids_s2 = []
            input_clicks_s2 = []
            
            for quid, uids, vids, true_clicks in batch_lines:
                qids_s2.append([quid])
                uids_s2.append(uids)
                vids_s2.append(vids)
                # Modelin beklediği 31 elemanlı "tıklama" girdisi
                # (modules.py'deki [:-1] dilimlemesi için)
                input_clicks_s2.append([0] + true_clicks) # [0, c_1, ..., c_30]
            
            # Modeli S2 için çalıştır (probs, rels, exams)
            probs_s2_tensor, _, _ = model(qids_s2, uids_s2, vids_s2, input_clicks_s2)
            
            # Sonuçları (quid, uid, rank) bazında aç
            for j in range(len(batch_lines)):
                quid, uids, _, _ = batch_lines[j]
                probs_s2_list_for_query = probs_s2_tensor[j].tolist()
                
                for k in range(max_d_num):
                    if uids[k] == 0: continue # Padding atla
                    results.append({
                        'quid': quid,
                        'uid': uids[k],
                        'rank': k + 1, # 1-bazlı rank
                        'graphcm_s2_contextual': probs_s2_list_for_query[k]
                    })

    print(f"-> S2 Tahminleri tamamlandı. {len(results)} adet (ilan) tahmini üretildi.")
    
    # 3. S2 sonuçlarını kaydet
    df_results_s2 = pd.DataFrame(results)
    df_results_s2.to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"\n--- BAŞARILI (S2) ---")
    print(f"Senaryo 2 tahminleri şu dosyaya kaydedildi: {OUTPUT_CSV_FILE}")
    print(df_results_s2.head().to_string())

if __name__ == "__main__":
    main()