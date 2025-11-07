# -*- coding: utf-8 -*-
"""
Eğitilmiş GraphCM modelini kullanarak belirli (sorgu, doküman) çiftleri için
tıklama olasılığı ve iç skorları (inceleme, çekicilik) tahmin eden script.

NOT: Bu script'in çalışması için GraphCM.py dosyasındaki 'forward' metodunun
'return pred_logits, rels, exams' olarak güncellenmesi gerekir.
"""

import torch
import argparse
import sys
import os
import numpy as np

# Projenin ana dizinini path'e ekleyerek Model ve Dataset'i import etmemizi sağlar
# Eğer script'i 'helpers' klasörü içinde çalıştırıyorsan bu satır kalmalı
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from dataset import Dataset

def predict(args, query_id, doc_id, position=1):
    """Modeli yükler, tek bir çift için girdiyi hazırlar ve tahmini yapar."""
    
    print("\n" + "="*50)
    print(f"TAHMİN BAŞLATILIYOR")
    print(f"  - Sorgu ID (quid): {query_id}")
    print(f"  - Doküman ID (uid): {doc_id}")
    print(f"  - Test Pozisyonu: {position}")
    print("="*50)

    # Cihazı belirle (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"-> Cihaz kullanılıyor: {device}")
    
    # Dataset objesini oluşturarak modelin ihtiyaç duyduğu bilgileri alıyoruz.
    print("-> Veri seti sözlükleri yükleniyor...")
    dataset = Dataset(args)
    
    # Modeli, eğitimde kullanılan parametrelerle birebir aynı şekilde oluşturuyoruz.
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    
    # Kaydedilmiş model ağırlıklarını yüklüyoruz.
    print(f"-> Model checkpoint'i yükleniyor: {args.model_dir} (Adım: {args.load_model})")
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    
    # Modeli GPU'ya veya CPU'ya taşıyoruz ve "değerlendirme" moduna alıyoruz.
    model.model.to(device)
    model.model.eval()

    # --- Girdi Verisini Hazırlama ---
    # Model, batch (liste içinde liste) bekler
    
    # 1. Sorgu ID'si
    qids = [[query_id]]
    
    # 2. Doküman ID'leri (Tüm pozisyonlar 0, sadece hedef pozisyon dolu)
    uids_list = [0] * args.max_d_num
    target_index = position - 1
    if 0 <= target_index < args.max_d_num:
        uids_list[target_index] = doc_id
    else:
        print(f"UYARI: Geçersiz pozisyon {position}. Pozisyon 1 (index 0) kullanılıyor.")
        uids_list[0] = doc_id
        target_index = 0
    uids = [uids_list]

    # 3. Vertical Tipler (Varsayılan olarak 1 kullanıyoruz)
    vids = [[1] * args.max_d_num]
    
    # 4. Tıklama Girdisi (Label Leakage düzeltmesine uygun olarak)
    # Model [0, c_1, ..., c_n-1] bekler. 
    # Tahmin yaparken geçmişi bilmediğimiz için [0, 0, ..., 0] yollarız.
    # Toplam 'max_d_num + 1' elemanlı olmalı.
    clicks = [[0] * (args.max_d_num + 1)]

    # --- Tahmin Yapma ve Sonucu Gösterme ---
    with torch.no_grad():
        # Model (pred_logits, rels, exams) döndürecek
        click_probabilities, attr_scores, exam_probs = model.model(qids, uids, vids, clicks)
    
    final_prediction = click_probabilities[0, target_index].item()
    exam_prediction = exam_probs[0, target_index].item()
    attr_prediction = attr_scores[0, target_index].item()

    print(f"\n--- SONUÇLAR (Pozisyon {position}) ---")
    print(f"  - P(Inceleme / Examination)   : {exam_prediction:.6f}")
    print(f"  - P(Çekicilik / Attractiveness): {attr_prediction:.6f}")
    print(f"  - Nihai Tıklama Olasılığı      : {final_prediction:.6f}")
    print("-" * 20)


if __name__ == "__main__":
    
    # --- Analiz edilecek 7 çifti burada tanımlıyoruz ---
    pairs_to_analyze = [
        {'name': 'Popüler, İyi CTR (24, 1389)', 'query_id': 24, 'doc_id': 1389},
        {'name': 'Popüler, İyi CTR (24, 1386)', 'query_id': 24, 'doc_id': 1386},
        {'name': 'Popüler, Düşük CTR (24, 326)', 'query_id': 24, 'doc_id': 326},
        {'name': 'Popüler, Çok Düşük CTR (44, 20335)', 'query_id': 44, 'doc_id': 20335},
        {'name': 'Popüler, İyi CTR (24, 3689)', 'query_id': 24, 'doc_id': 3689},
        {'name': 'Az Popüler, Çok Yüksek CTR (1475, 22840)', 'query_id': 1475, 'doc_id': 22840},
        {'name': 'Az Popüler, Yüksek CTR (4993, 12005)', 'query_id': 4993, 'doc_id': 12005},
    ]

    # --- Modelin eğitildiği sabit (hardcoded) parametreler ---
    model_params = {
        'dataset': 'emj_train_test',
        'model_dir': './outputs/models/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15', # GÜNCELLENDİ
        'result_dir': './outputs/results/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15', # GÜNCELLENDİ
        'summary_dir': './outputs/summary/GraphCM_emj_20epoch_lr0003_drop04_gnn_sample15', # GÜNCELLENDİ
        'log_dir': './outputs/log/',
        'algo': 'GraphCM',
        'optim': 'adam',
        'learning_rate': 0.0003, # GÜNCELLENDİ
        'batch_size': 512,
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
        'dropout_rate': 0.4, # GÜNCELLENDİ
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
        'use_pretrain_embed': False,
        'train': False, 'valid': False, 'test': False, 'rank': False,
    }

    # Sabit parametreleri bir Namespace objesine dönüştür
    model_args = argparse.Namespace(**model_params)
    
    # Yüklenecek checkpoint'i belirle 
    model_args.load_model = 21150
    
    # 'best' checkpoint'i yüklemek istersen:
    # model_args.load_model = 'best' # (Eğer .save_model() 'best' olarak kaydediyorsa)

    # Seçilen her bir çift için analizi çalıştır
    for pair in pairs_to_analyze:
        print(f"\n\n{'='*20} {pair['name']} {'='*20}")
        
        # Varsayılan olarak Pozisyon 1'de tahmin yap
        predict(model_args, pair['query_id'], pair['doc_id'], position=1)
        
        # Diğer pozisyonlarda da tahmin yapılabilir
        # predict(model_args, pair['query_id'], pair['doc_id'], position=5)
        # predict(model_args, pair['query_id'], pair['doc_id'], position=10)