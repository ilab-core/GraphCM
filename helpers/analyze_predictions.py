# !/usr/bin/python
# coding: utf8
# analyze_predictions.py

import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm

from dataset import Dataset
from model import Model
from utils import check_path

def parse_args():
    """
    Komut satırı argümanlarını okur. run.py ile TAM UYUMLU hale getirildi.
    """
    parser = argparse.ArgumentParser('GraphCM Prediction Analyzer')
    parser.add_argument('--dataset', required=True, help='Analiz edilecek veri setinin adı')
    parser.add_argument('--model_dir', required=True, help='Kaydedilmiş modellerin bulunduğu klasör')
    parser.add_argument('--load_model', type=int, required=True, help='Yüklenecek modelin step numarası')
    parser.add_argument('--algo', default='GraphCM')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--max_d_num', type=int, default=30)
    parser.add_argument('--embed_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--click_embed_size', type=int, default=4)
    parser.add_argument('--vtype_embed_size', type=int, default=8)
    parser.add_argument('--pos_embed_size', type=int, default=4)
    parser.add_argument('--combine', default='mul')
    parser.add_argument('--use_gnn', action='store_true', default=True)
    parser.add_argument('--gnn_att_heads', type=int, default=2)
    parser.add_argument('--gnn_neigh_sample', type=int, default=10)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--gnn_concat', action='store_true', default=False)
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    parser.add_argument('--gnn_leaky_slope', type=float, default=0.2)
    parser.add_argument('--inter_leaky_slope', type=float, default=0.2)
    parser.add_argument('--inter_neigh_sample', type=int, default=0)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.99)
    parser.add_argument('--reg_relevance', type=float, default=1.0)
    parser.add_argument('--use_pretrain_embed', action='store_true', default=False)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.3) 
    parser.add_argument('--data_parallel', action='store_true', default=False) 
    parser.add_argument('--learning_rate', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=1e-05)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--result_dir', default='./outputs/results/temp_analyzer')
    parser.add_argument('--summary_dir', default='./outputs/summary/temp_analyzer')
    
    return parser.parse_args()

def run_analysis(args):
    """
    Analiz sürecini yönetir.
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    
    print(f"Cihaz: {device}")

    # 1. Veri setini yükle
    print(f"'{args.dataset}' veri seti yükleniyor...")
    dataset = Dataset(args)

    # 2. Modeli hazırla ve eğitilmiş ağırlıkları yükle
    print("Model hazırlanıyor...")
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    
    print(f"'{args.load_model}' adımlı model yükleniyor...")
    try:
        model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    except FileNotFoundError:
        print("\nHATA: Model dosyası bulunamadı!")
        print(f"Kontrol edin: '{os.path.join(args.model_dir, args.algo + '_' + str(args.load_model) + '.model')}'")
        return

    # 3. Analiz için listeleri hazırla
    clicked_preds = []
    non_clicked_preds = []

    # 4. Test seti üzerinde tahmin yap
    print("Test seti üzerinde tahminler yapılıyor...")
    test_batches = dataset.gen_mini_batches('test', args.batch_size, shuffle=False)
    
    model.model.eval() # Modeli değerlendirme moduna al
    with torch.no_grad(): # Gradient hesaplamasını kapatarak hızlan
        for batch in tqdm(test_batches, desc="Batch'ler işleniyor"):
            
            # Gerekli verileri tensörlere çevir
            TRUE_CLICKS = torch.tensor(np.array(batch['true_clicks']), dtype=torch.float32).to(device)
            MASK = torch.tensor(np.array(batch['masks']), dtype=torch.bool).to(device)

            # Modelden tahminleri al
            pred_logits, _ = model.model(batch['qids'], batch['uids'], batch['vids'], batch['clicks'])
            
            # Padding'li kısımları maske ile ele
            masked_preds = torch.masked_select(pred_logits, MASK)
            masked_true_labels = torch.masked_select(TRUE_CLICKS, MASK)
            
            # Sonuçları ilgili listelere ekle
            for pred, true_label in zip(masked_preds, masked_true_labels):
                if true_label.item() == 1:
                    clicked_preds.append(pred.item())
                else:
                    non_clicked_preds.append(pred.item())
                    
    # 5. Sonuçları analiz et ve raporla
    print("\n" + "="*50)
    print("ANALİZ SONUÇLARI")
    print("="*50)

    if clicked_preds:
        clicked_preds = np.array(clicked_preds)
        print(f"\n--- GERÇEKTE TIKLANAN ({len(clicked_preds)} adet) İLANLAR İÇİN TAHMİNLER ---")
        print(f"  Ortalama Tıklama Olasılığı: {clicked_preds.mean():.6f}")
        print(f"  Standart Sapma            : {clicked_preds.std():.6f}")
        print(f"  Minimum Tahmin Olasılığı  : {clicked_preds.min():.6f}")
        print(f"  Maksimum Tahmin Olasılığı : {clicked_preds.max():.6f}")
    else:
        print("\n--- GERÇEKTE TIKLANAN İLAN BULUNAMADI ---")
        
    if non_clicked_preds:
        non_clicked_preds = np.array(non_clicked_preds)
        print(f"\n--- GERÇEKTE TIKLANMAYAN ({len(non_clicked_preds)} adet) İLANLAR İÇİN TAHMİNLER ---")
        print(f"  Ortalama Tıklama Olasılığı: {non_clicked_preds.mean():.6f}")
        print(f"  Standart Sapma            : {non_clicked_preds.std():.6f}")
        print(f"  Minimum Tahmin Olasılığı  : {non_clicked_preds.min():.6f}")
        print(f"  Maksimum Tahmin Olasılığı : {non_clicked_preds.max():.6f}")
    else:
        print("\n--- GERÇEKTE TIKLANMAYAN İLAN BULUNAMADI ---")
        
    print("\n" + "="*50)

if __name__ == '__main__':
    args = parse_args()
    check_path(args.summary_dir) 
    run_analysis(args)