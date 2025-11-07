# -*- coding: utf-8 -*-
"""
Test setindeki her bir ilan için Senaryo 2 (Bağlamsal) 
tahminini üretir.

Üretilen her tahmini, o ilanın *gerçek* tıklanma durumuna 
(tıklandı / tıklanmadı) göre sınıflandırır ve tek bir CSV olarak kaydeder.
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

# Model parametreleri (Checkpoint: 21150)
MODEL_PARAMS = {
    'dataset': 'emj_train_test',
    'model_dir': './outputs/models/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'algo': 'GraphCM', 'learning_rate': 0.0003, 'batch_size': 1024, # Tahmin için batch_size
    'embed_size': 32, 'hidden_size': 64, 'max_d_num': 30, 'vtype_embed_size': 8,
    'click_embed_size': 4, 'pos_embed_size': 4, 'combine': 'mul', 'use_gnn': True,
    'gnn_neigh_sample': 15, 'gnn_att_heads': 2, 'weight_decay': 1e-05, 'dropout_rate': 0.4,
    'load_model': 21150, 'optim': 'adam', 'momentum': 0.99, 'gnn_dropout': 0, 
    'gnn_leaky_slope': 0.2, 'gnn_concat': False, 'inter_neigh_sample': 0, 
    'inter_leaky_slope': 0.2, 'gpu_num': 1, 'data_parallel': False, 'num_steps': 28200, 
    'eval_freq': 1410, 'check_point': 1410, 'patience': 5, 'lr_decay': 0.5, 'num_iter': 1, 
    'reg_relevance': 1.0, 'use_pretrain_embed': False, 'train': False, 'valid': False, 
    'test': False, 'rank': False,
    'result_dir': './outputs/results/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'summary_dir': './outputs/summary/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'log_dir': './outputs/log/',
}

# --- Girdi ve Çıktı Dosyaları ---
OUR_TEST_FILE = 'data/emj_train_test/test_per_query_quid.txt' # Girdi (click bilgisi burada)
OUTPUT_CSV_FILE = 'compare/click_vs_noclick_probs.csv' # Çıktı

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
    print("GraphCM Tıklandı/Tıklanmadı Analizi Başlatıldı.")
    
    model_args = argparse.Namespace(**MODEL_PARAMS)
    model, device = load_model_and_device(model_args)
    max_d_num = model_args.max_d_num
    
    # Sonuçları `clicks` ve `no_clicks` için burada biriktireceğiz
    results = [] # [{'prob': 0.123, 'is_click': 1}, {'prob': 0.045, 'is_click': 0}, ...]
    batch_size = model_args.batch_size
    
    try:
        with open(OUR_TEST_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"HATA: Bizim test dosyamız bulunamadı: {OUR_TEST_FILE}")
        return

    print(f"-> {len(lines)} sorgu (satır) bizim test setimizden işlenecek...")
    
    all_parsed_lines = []
    for line in lines:
        parsed = parse_our_test_line(line, max_d_num)
        if parsed:
            all_parsed_lines.append(parsed)

    with torch.no_grad():
        for i in tqdm(range(0, len(all_parsed_lines), batch_size), desc="Bağlamsal Tahminler Hesaplanıyor"):
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
                input_clicks_s2.append([0] + true_clicks) # [0, c_1, ..., c_30]
            
            # Modeli S2 için çalıştır (probs, rels, exams)
            probs_s2_tensor, _, _ = model(qids_s2, uids_s2, vids_s2, input_clicks_s2)
            
            # Sonuçları (prob) ve (is_click) olarak ayır
            for j in range(len(batch_lines)):
                _, uids, _, true_clicks = batch_lines[j] # Gerçek tıklama listesi
                probs_s2_list_for_query = probs_s2_tensor[j] # Tahmin listesi
                
                for k in range(max_d_num):
                    if uids[k] == 0: continue # Padding atla
                    
                    prob = probs_s2_list_for_query[k].item()
                    actual_click = true_clicks[k]
                    
                    results.append({
                        'probability': prob, # Tahmin edilen olasılık
                        'is_click': actual_click # Gerçek tıklanma durumu (1 veya 0)
                    })

    print(f"-> Analiz tamamlandı. {len(results)} adet (ilan) tahmini üretildi.")
    
    # Sonuçları CSV'ye kaydet
    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"\n--- BAŞARILI ---")
    print(f"Tıklandı/Tıklanmadı tahminleri şu dosyaya kaydedildi: {OUTPUT_CSV_FILE}")
    print(df_results.head().to_string())

if __name__ == "__main__":
    main()