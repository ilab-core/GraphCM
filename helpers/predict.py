# -*- coding: utf-8 -*-

"""
Eğitilmiş GraphCM modelini kullanarak tek bir (sorgu, doküman) çifti için
tıklama olasılığı tahmini yapan script.

Örnek Komut:
python helpers/predict.py --query_id 1333 --doc_id 14139
"""

# 1. Gerekli Kütüphaneler ve Path Düzeltmesi
import torch
import argparse
import sys
import os

# Projenin ana dizinini path'e ekleyerek Model ve Dataset'i import etmemizi sağlar
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from dataset import Dataset

# 2. Ana Tahmin Fonksiyonu
def predict(args):
    """Modeli yükler, girdiyi hazırlar ve tahmini yapar."""

    print("Model ve veri yükleniyor...")
    
    # Cihazı belirle (GPU varsa kullan, yoksa CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    # Dataset objesini oluşturarak modelin ihtiyaç duyduğu bilgileri (query/doc sayısı vb.) alıyoruz.
    dataset = Dataset(args)
    
    # Modeli, eğitimde kullanılan parametrelerle birebir aynı şekilde oluşturuyoruz.
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    
    # Kaydedilmiş model ağırlıklarını yüklüyoruz.
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    
    # Modeli GPU'ya veya CPU'ya taşıyoruz.
    model.model.to(device)
    
    # Modeli "değerlendirme" moduna alıyoruz. Bu, dropout gibi katmanları devre dışı bırakır.
    model.model.eval()

    print("-" * 30)
    print(f"Tahmin yapılıyor: Sorgu ID = {args.query_id}, Doküman ID = {args.doc_id}")
    
    # 3. Modelin Girdisini Hazırlama
    # ----------------------------------------------------
    query_id = args.query_id
    doc_id = args.doc_id
    
    # Sorguyu, modelin aynı anda bir grup veriyi işleme formatına uygun hale getiriyoruz.
    qids = torch.LongTensor([[query_id]]).to(device)
    # Model 10'luk liste beklediği için, test edeceğimiz dokümanı ilk sıraya koyup gerisini boş bırakıyoruz.
    uids = torch.LongTensor([[doc_id] + [0] * 9]).to(device)
    #uids = torch.LongTensor([[0] * 4 + [doc_id] + [0] * 5]).to(device)
    # Doküman tipi (vtype) bilgimiz olmasa da, eğitimdekiyle uyumlu olması için standart '1' değeriyle dolduruyoruz.
    vids = torch.LongTensor([[1] * 10]).to(device)
    # Gelecekteki bir tıklamayı tahmin ettiğimiz için, "henüz tıklama olmadı" durumunu temsil eden nötr girdi gönderiyoruz.
    clicks = torch.LongTensor([[0] * 10]).to(device)

    # 4. Tahmin Yapma ve Sonucu Gösterme
    # ----------------------------------------------------
    with torch.no_grad(): # Gradyan hesaplamasını kapatarak süreci hızlandırıyoruz.
        # Modeli çalıştırıp tıklama olasılıklarını alıyoruz.
        click_probabilities, exam_probs, attr_scores = model.model(qids, uids, vids, clicks)
    
    # İlgilendiğimiz ilk dokümanın olasılığını alıyoruz.
    final_prediction = click_probabilities[0, 0].item()
    exam_prediction = exam_probs[0, 0].item()
    attr_prediction = attr_scores[0, 0].item()

    print("\n" + "="*40)
    print(f"SONUÇ: Modelin İç Değerleri:")
    print(f"  - P(Inceleme / Examination): {exam_prediction:.4f}")
    print(f"  - P(Çekicilik / Attractiveness): {attr_prediction:.4f}")
    print(f"  - Nihai Tıklama Olasılığı (E * A): {final_prediction:.4f}")
    print("="*40)


# 5. Script'i Çalıştırma Bloğu
# ----------------------------------------------------
if __name__ == "__main__":
    
    # 1. Komut satırından alınacak değişken argümanları tanımla
    parser = argparse.ArgumentParser(description='GraphCM ile tekil tahmin yapma scripti.')
    parser.add_argument('--query_id', type=int, required=True, help='Tahmin yapılacak sorgunun IDsi.')
    parser.add_argument('--doc_id', type=int, required=True, help='Tahmin yapılacak dokümanın IDsi.')
    parser.add_argument('--load_model', type=int, default=15492, help='Yüklenecek modelin adım numarası (checkpoint). Varsayılan: Son epoch.')
    script_args = parser.parse_args()

    # 2. Modelin eğitildiği sabit (hardcoded) parametreleri bir sözlükte topla
    # Bu değerler "expC_neigh15_lr_0_0005" deneyinden alınmıştır.
    model_params = {
        # --- Temel Ayarlar ---
        'dataset': 'emj',
        'model_dir': './outputs/models/expD_neigh20_lr_0_0005',      # DÜZELTİLDİ
        'result_dir': './outputs/results/expD_neigh20_lr_0_0005',
        'summary_dir': './outputs/summary/expD_neigh20_lr_0_0005',
        'log_dir': './outputs/log/',
        'algo': 'GraphCM',
        
        # --- Mimari ve Optimizer Ayarları ---
        'batch_size': 512,
        'optim': 'adam',
        'learning_rate': 0.0005,
        'embed_size': 32,
        'hidden_size': 64,
        'vtype_embed_size': 8,
        'click_embed_size': 4,
        'pos_embed_size': 4,
        'combine': 'mul',
        'use_gnn': True,
        'gnn_att_heads': 2,
        'weight_decay': 1e-05,
        'momentum': 0.99,
        'dropout_rate': 0.5,
        'gnn_neigh_sample': 20,
        
        # --- Diğer Zorunlu Parametreler ---
        'max_d_num': 10,
        'gnn_dropout': 0,
        'gnn_leaky_slope': 0.2,
        'gnn_concat': False,
        'inter_neigh_sample': 0,
        'inter_leaky_slope': 0.2,
        'gpu_num': 1,
        'data_parallel': False,
        'eval_freq': 1291,
        'check_point': 1291,
        'patience': 5,
        'lr_decay': 0.5,
        'train': False, 'valid': False, 'test': False, 'rank': False,
        'num_iter': 1,
        'reg_relevance': 1.0,
        'use_pretrain_embed': False
    }

    # 3. Sabit parametreleri ve komut satırı argümanlarını birleştir
    model_args = argparse.Namespace(**model_params)
    model_args.query_id = script_args.query_id
    model_args.doc_id = script_args.doc_id
    model_args.load_model = script_args.load_model

    # 4. Ana tahmin fonksiyonunu çalıştır
    predict(model_args)
