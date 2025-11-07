# -*- coding: utf-8 -*-
"""
Hacer'in IDBN modelinin çıktılarını (CSV) okur,
GraphCM modelinin anladığı ID'lere (quid, uid) eşler,
tüm veri seti için toplu (batch) tahmin yapar ve
GraphCM tahminlerini de içeren yeni bir CSV dosyası kaydeder.
"""

import torch
import argparse
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from dataset import Dataset
import utils  # utils.py'nin ana dizinde olduğunu varsayıyoruz

# --- Modelin Eğitildiği Parametreler ---
# Kullandığımız 'lr0003_drop04_gnn_sample15' modelinin ayarları
MODEL_PARAMS = {
    'dataset': 'emj_train_test',
    'model_dir': './outputs/models/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'result_dir': './outputs/results/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'summary_dir': './outputs/summary/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15',
    'log_dir': './outputs/log/',
    'algo': 'GraphCM',
    'optim': 'adam',
    'learning_rate': 0.0003,
    'batch_size': 512, # Tahmin için batch_size'ı yüksek tutabiliriz
    'embed_size': 32,
    'hidden_size': 64,
    'max_d_num': 30,
    'vtype_embed_size': 8,
    'click_embed_size': 4,
    'pos_embed_size': 4,
    'combine': 'mul',
    'use_gnn': True,
    'gnn_neigh_sample': 15,
    'gnn_att_heads': 2,
    'weight_decay': 1e-05,
    'dropout_rate': 0.4,
    'momentum': 0.99,
    'gnn_dropout': 0, 'gnn_leaky_slope': 0.2, 'gnn_concat': False,
    'inter_neigh_sample': 0, 'inter_leaky_slope': 0.2, 'gpu_num': 1,
    'data_parallel': False,
    'num_steps': 28200,
    'eval_freq': 1410,
    'check_point': 1410,
    'patience': 5,
    'lr_decay': 0.5,
    'num_iter': 1,
    'reg_relevance': 1.0,
    'load_model': 21150,
    'use_pretrain_embed': False,
    'train': False, 'valid': False, 'test': False, 'rank': False,
}


# Hacer'in Dosyaları
INPUT_FILE = 'compare/IDBN_click_probs.csv' 
OUTPUT_FILE = 'compare/IDBN_click_probs_with_GraphCM.csv'


def map_ids(df, data_dir):
    """Raw ID'leri (queryid, item_id) GraphCM ID'lerine (quid, uid) map'ler."""
    print("-> ID'ler eşleştiriliyor...")
    
    # Sözlükleri yükle
    try:
        query_qid_dict = utils.load_dict(data_dir, 'query_qid.dict')
        url_uid_dict = utils.load_dict(data_dir, 'url_uid.dict')
    except FileNotFoundError as e:
        print(f"HATA: Sözlük dosyaları bulunamadı: {e}")
        print(f"Lütfen '{data_dir}' yolunun doğru olduğundan emin olun.")
        return None
        
    # Pandas map fonksiyonunu kullanarak eşleştir
    # NOT: Hacer'in ID'leri str olabilir, bizimkiler int. Her ikisini de str'ye çevirerek eşleştirmek en güvenlisi.
    # (Eğer ID'leriniz zaten int ise .astype(str) kısımları kaldırılabilir)
    
    # Önce dict'lerin key'lerini str yap
    query_qid_dict_str = {str(k): v for k, v in query_qid_dict.items()}
    url_uid_dict_str = {str(k): v for k, v in url_uid_dict.items()}

    df['quid'] = df['queryid'].astype(str).map(query_qid_dict_str)
    df['uid'] = df['item_id'].astype(str).map(url_uid_dict_str)
    
    # Eşleşmeyen ID'leri kontrol et
    original_rows = len(df)
    df.dropna(subset=['quid', 'uid'], inplace=True)
    mapped_rows = len(df)
    
    if mapped_rows < original_rows:
        print(f"UYARI: {original_rows - mapped_rows} satır eşleşmeyen ID nedeniyle atıldı (cold-start).")
        
    # Tipleri modele uygun hale getir
    df['quid'] = df['quid'].astype(int)
    df['uid'] = df['uid'].astype(int)
    df['rank'] = df['rank'].astype(int) #
    
    print(f"-> Eşleştirme tamamlandı. {mapped_rows} satır işlenecek.")
    return df

def batch_predict(df, args):
    """Modeli yükler ve DataFrame'deki tüm satırlar için toplu tahmin yapar."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Cihaz kullanılıyor: {device}")

    # Dataset ve Model'i yükle
    print("-> Veri seti sözlükleri ve model yapısı yükleniyor...")
    dataset = Dataset(args)
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    
    print(f"-> Model checkpoint'i yükleniyor (Adım: {args.load_model})...")
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    
    model.model.to(device)
    model.model.eval()

    predictions = []
    batch_size = args.batch_size
    max_d_num = args.max_d_num

    print(f"-> {len(df)} satır için toplu tahmin başlatılıyor (Batch Size: {batch_size})...")
    
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Batch'ler işleniyor"):
            end_idx = start_idx + batch_size
            batch_df = df.iloc[start_idx:end_idx]
            
            # Model girdilerini hazırla
            qids_list = []
            uids_list = []
            vids_list = []
            clicks_list = []
            
            for _, row in batch_df.iterrows():
                qids_list.append([row['quid']])
                
                # uids: Sadece 'rank' pozisyonu 'uid' ile dolu, kalanı 0
                uids_row = [0] * max_d_num
                rank_index = row['rank'] - 1 # 1-bazlı 'rank'ı 0-bazlı index'e çevir
                if 0 <= rank_index < max_d_num:
                    uids_row[rank_index] = row['uid']
                uids_list.append(uids_row)
                
                vids_list.append([1] * max_d_num) # vtype=1 varsay
                clicks_list.append([0] * (max_d_num + 1)) # Geçmiş tıklama yok
            
            # Tahmin yap
            # Not: Model.model (GraphCM) 3 değer döndürecek şekilde güncellenmişti
            click_probs, attr_scores, exam_probs = model.model(
                qids_list, uids_list, vids_list, clicks_list
            )
            
            # Çıktıdan doğru rank'taki tahmini çek
            for i, (_, row) in enumerate(batch_df.iterrows()):
                rank_index = row['rank'] - 1
                if 0 <= rank_index < max_d_num:
                    pred_value = click_probs[i, rank_index].item()
                    predictions.append(pred_value)
                else:
                    predictions.append(None) # Geçersiz rank durumu

    print("-> Tahmin tamamlandı.")
    df['graphcm_prob'] = predictions
    return df

def main():
    print("Karşılaştırma Script'i Başlatıldı.")
    
    # 1. Model ayarlarını Namespace'e çevir
    model_args = argparse.Namespace(**MODEL_PARAMS)
    
    # 2. Hacer'in verisini yükle
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"HATA: Girdi dosyası bulunamadı: {INPUT_FILE}")
        return
    
    # 3. ID'leri eşleştir
    data_dir = os.path.join('data', model_args.dataset)
    df_mapped = map_ids(df, data_dir)
    
    if df_mapped is None:
        print("Eşleştirme hatası nedeniyle script durduruldu.")
        return
        
    # 4. Toplu tahmin yap
    df_final = batch_predict(df_mapped, model_args)
    
    # 5. Sonuçları kaydet
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"-> Başarılı! Sonuçlar şu dosyaya kaydedildi: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()