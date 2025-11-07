# -*- coding: utf-8 -*-
"""
Senaryo 1 (Bağımsız Çekicilik) için tahminler üretir.
'full_click_prob' sütunu ile karşılaştırılacaktır.

IDBN CSV'sini okur, (quid, uid) eşleştirmesi yapar ve 
her (q, u) çiftini rank=1 / sıfır-geçmiş varsayımıyla tahmin eder.
"""

import torch
import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import Model
from dataset import Dataset
import utils

# Model parametreleri (Checkpoint: 21150)
MODEL_PARAMS = {
    'dataset': 'emj_train_test',
    'model_dir': './outputs/models/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15', 'optim': 'adam',
    'algo': 'GraphCM', 'learning_rate': 0.0003, 'batch_size': 1024, # Tahmin için batch_size artırıldı
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
IDBN_CSV_FILE = 'compare/IDBN_click_probs.csv' # Girdi
OUTPUT_CSV_FILE = 'compare/GraphCM_S1_Rank1_Predictions.csv' # Çıktı

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

def load_and_map_idbn_data(idbn_file_path, data_dir):
    """IDBN CSV'sini okur ve ID'leri quid/uid'ye map'ler."""
    print(f"-> IDBN dosyası okunuyor: {idbn_file_path}")
    df_idbn = pd.read_csv(idbn_file_path)

    print("-> ID'ler eşleştiriliyor...")
    query_qid_dict = utils.load_dict(data_dir, 'query_qid.dict')
    url_uid_dict = utils.load_dict(data_dir, 'url_uid.dict')
    query_qid_dict_str = {str(k): v for k, v in query_qid_dict.items()}
    url_uid_dict_str = {str(k): v for k, v in url_uid_dict.items()}

    df_idbn['quid'] = df_idbn['queryid'].astype(str).map(query_qid_dict_str)
    df_idbn['uid'] = df_idbn['item_id'].astype(str).map(url_uid_dict_str)
    
    original_rows = len(df_idbn)
    df_idbn.dropna(subset=['quid', 'uid'], inplace=True)
    print(f"   {len(df_idbn)}/{original_rows} satır ID'lerle eşleşti.")
    
    df_idbn['quid'] = df_idbn['quid'].astype(int)
    df_idbn['uid'] = df_idbn['uid'].astype(int)
    df_idbn['rank'] = df_idbn['rank'].astype(int) # Orijinal rank'ı koru (merge için)
    
    # S1 için (q, u) çiftleri benzersiz olmalı, rank'tan bağımsız
    df_unique_pairs = df_idbn[['quid', 'uid']].drop_duplicates()
    print(f"   {len(df_unique_pairs)} adet benzersiz (quid, uid) çifti bulundu.")
    return df_idbn, df_unique_pairs

def main():
    print("GraphCM Senaryo 1 (Rank=1) Tahmin Script'i Başlatıldı.")
    
    model_args = argparse.Namespace(**MODEL_PARAMS)
    model, device = load_model_and_device(model_args)
    max_d_num = model_args.max_d_num
    
    # 1. IDBN verisini yükle, map'le ve benzersiz (q,u) çiftlerini al
    df_idbn, df_unique_pairs = load_and_map_idbn_data(IDBN_CSV_FILE, os.path.join('data', model_args.dataset))
    
    results_s1 = [] # (quid, uid, prob_s1)
    batch_size = model_args.batch_size
    
    print(f"-> {len(df_unique_pairs)} benzersiz (q, u) çifti için Senaryo 1 tahminleri yapılıyor...")

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(df_unique_pairs), batch_size), desc="S1 Batch'leri işleniyor"):
            end_idx = start_idx + batch_size
            batch_df = df_unique_pairs.iloc[start_idx:end_idx]
            
            # Girdileri hazırla (batch_size, max_d_num)
            qids_list = []
            uids_list = []
            vids_list = [[1] * max_d_num] * len(batch_df) # Sabit vtype
            clicks_list = [[0] * (max_d_num + 1)] * len(batch_df) # Sıfır geçmiş
            
            for _, row in batch_df.iterrows():
                qids_list.append([row['quid']])
                
                # uids: Sadece pozisyon 1 (index 0) dolu
                uids_row = [0] * max_d_num
                uids_row[0] = row['uid'] # ilanı 1. pozisyona koy
                uids_list.append(uids_row)
            
            # Modeli S1 için çalıştır (probs, rels, exams)
            probs_s1_tensor, _, _ = model(qids_list, uids_list, vids_list, clicks_list)
            
            # Her satırın SADECE 1. pozisyonundaki (index 0) tahmini al
            probs_s1_batch_list = probs_s1_tensor[:, 0].tolist() 
            
            # Sonuçları kaydet
            for i, prob in enumerate(probs_s1_batch_list):
                results_s1.append({
                    'quid': batch_df.iloc[i]['quid'],
                    'uid': batch_df.iloc[i]['uid'],
                    'graphcm_s1_rank1': prob
                })

    print("-> S1 Tahminleri tamamlandı.")
    
    # 2. S1 sonuçlarını ana IDBN tablosuyla birleştir
    df_results_s1 = pd.DataFrame(results_s1)
    
    df_final = pd.merge(
        df_idbn, 
        df_results_s1, 
        on=['quid', 'uid'], # (q, u) bazında birleştir
        how='left' # IDBN dosyasındaki tüm satırları koru
    )
    
    # Sadece S1 için gerekli sütunları kaydet (S2 sonra eklenecek)
    output_cols = ['quid', 'uid', 'rank', 'full_click_prob', 'conditional_click_prob', 'graphcm_s1_rank1']
    df_final[output_cols].to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"\n--- BAŞARILI (S1) ---")
    print(f"Senaryo 1 tahminleri şu dosyaya kaydedildi: {OUTPUT_CSV_FILE}")
    print(df_final.head().to_string())

if __name__ == "__main__":
    main()