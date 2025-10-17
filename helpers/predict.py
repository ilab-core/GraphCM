# -*- coding: utf-8 -*-
"""
Eğitilmiş GraphCM modelini kullanarak belirli (sorgu, doküman) çiftleri için
tıklama olasılığı ve iç skorları (inceleme, çekicilik) tahmin eden script.
"""

import torch
import argparse
import sys
import os
import numpy as np

# Projenin ana dizinini path'e ekleyerek Model ve Dataset'i import etmemizi sağlar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from dataset import Dataset

def predict(args, query_id, doc_id, position=1):
    """Modeli yükler, tek bir çift için girdiyi hazırlar ve tahmini yapar."""
    
    print("\n" + "="*50)
    print(f"TAHMİN BAŞLATILIYOR")
    print(f"  - Sorgu ID: {query_id}")
    print(f"  - Doküman ID: {doc_id}")
    print(f"  - Test Pozisyonu: {position}")
    print("="*50)

    # Cihazı belirle (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset objesini oluşturarak modelin ihtiyaç duyduğu bilgileri alıyoruz.
    dataset = Dataset(args)
    
    # Modeli, eğitimde kullanılan parametrelerle birebir aynı şekilde oluşturuyoruz.
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    
    # Kaydedilmiş model ağırlıklarını yüklüyoruz.
    print(f"-> Model checkpoint'i yükleniyor: {args.load_model}")
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    
    # Modeli GPU'ya veya CPU'ya taşıyoruz ve "değerlendirme" moduna alıyoruz.
    model.model.to(device)
    model.model.eval()

    # Tensörleri en başta oluşturmak yerine, tıpkı dataset.py'deki gibi
    # Python listeleri olarak hazırlıyoruz.
    qids = [[query_id]]
    
    uids_list = [0] * args.max_d_num
    if 1 <= position <= args.max_d_num:
        uids_list[position - 1] = doc_id
    else:
        uids_list[0] = doc_id
    uids = [uids_list]

    vids = [[1] * args.max_d_num]
    
    # clicks listesi, dataset.py'deki gibi 1 (placeholder) + 30 (gerçek) = 31 elemanlı olmalı
    clicks = [[0] * (args.max_d_num + 1)]

    # --- Tahmin Yapma ve Sonucu Gösterme ---
    with torch.no_grad():
        click_probabilities, exam_probs, attr_scores = model.model(qids, uids, vids, clicks)
    
    target_index = position - 1
    final_prediction = click_probabilities[0, target_index].item()
    exam_prediction = exam_probs[0, target_index].item()
    attr_prediction = attr_scores[0, target_index].item()

    print(f"\n--- SONUÇLAR ---")
    print(f"  - P(Inceleme / Examination)  : {exam_prediction:.4f}")
    print(f"  - P(Çekicilik / Attractiveness): {attr_prediction:.4f}")
    print(f"  - Nihai Tıklama Olasılığı      : {final_prediction:.4f}")
    print("-" * 20)


if __name__ == "__main__":
    
    # --- Analiz edilecek 5 çifti burada tanımlıyoruz ---
    pairs_to_analyze = [
        {'name': '"İyi" Örnek (Yüksek CTR)', 'query_id': 1475, 'doc_id': 22820},
        {'name': '"Kötü" Örnek (Sıfır CTR)', 'query_id': 44, 'doc_id': 40},
        {'name': '"Orta-İyi" Örnek', 'query_id': 648, 'doc_id': 12083},
        {'name': '"İyi" Örnek 2 (Tutarlılık)', 'query_id': 4377, 'doc_id': 11977},
        {'name': '"Kötü" Örnek 2 (Tutarlılık)', 'query_id': 271, 'doc_id': 40},
    ]

    # --- Modelin eğitildiği sabit (hardcoded) parametreler ---
    model_params = {
        'dataset': 'emj_30ilan',
        'model_dir': './outputs/models/final_run',
        'result_dir': './outputs/results/final_run',
        'summary_dir': './outputs/summary/final_run',
        'log_dir': './outputs/log/',
        'algo': 'GraphCM',
        'optim': 'adam',
        'learning_rate': 0.0005,
        'batch_size': 512,
        'embed_size': 32,
        'hidden_size': 64,
        'max_d_num': 30,
        'vtype_embed_size': 8,
        'click_embed_size': 4,
        'pos_embed_size': 4,
        'combine': 'mul',
        'use_gnn': True,
        'gnn_neigh_sample': 10,
        'gnn_att_heads': 2,
        'weight_decay': 1e-05,
        'dropout_rate': 0.5,
        'momentum': 0.99,
        'gnn_dropout': 0, 'gnn_leaky_slope': 0.2, 'gnn_concat': False,
        'inter_neigh_sample': 0, 'inter_leaky_slope': 0.2, 'gpu_num': 1,
        'data_parallel': False,
        'num_steps': 14100,
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
    
    # Yüklenecek en iyi modeli belirliyoruz 
    model_args.load_model = 14100 

    # Seçilen her bir çift için analizi çalıştır
    for pair in pairs_to_analyze:
        print(f"\n\n{'='*20} {pair['name']} {'='*20}")
        model_args.query_id = pair['query_id']
        model_args.doc_id = pair['doc_id']
        
        # Ana tahmin fonksiyonunu çalıştır
        predict(model_args, model_args.query_id, model_args.doc_id)