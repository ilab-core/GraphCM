# helpers/rerank.py

import torch
import argparse
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Model
from dataset import Dataset

def rerank(args):
    """Modeli yükler, bir doküman listesi alır ve her biri için tahmin yapar."""

    print("Model ve veri yükleniyor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    dataset = Dataset(args)
    model = Model(args, dataset.query_size, dataset.doc_size, dataset.vtype_size, dataset)
    model.load_model(model_dir=args.model_dir, model_prefix=args.algo, global_step=args.load_model)
    model.model.to(device)
    model.model.eval()

    print("-" * 30)
    print(f"Yeniden sıralama yapılıyor: Sorgu ID = {args.query_id}")

    # Girdiyi hazırla
    query_id = args.query_id
    # Komut satırından gelen string listeyi Python listesine çevir
    doc_ids = json.loads(args.doc_ids)

    if len(doc_ids) != 10:
        print(f"HATA: Doküman listesi tam olarak 10 elemanlı olmalıdır. Gelen liste: {len(doc_ids)} elemanlı.")
        return

    qids = torch.LongTensor([[query_id]]).to(device)
    uids = torch.LongTensor([doc_ids]).to(device) # Gelen 10'luk listeyi doğrudan kullan
    vids = torch.LongTensor([[1] * 10]).to(device)
    clicks = torch.LongTensor([[0] * 10]).to(device)

    # Tahmin yap
    with torch.no_grad():
        click_probabilities, exam_probs, attr_scores = model.model(qids, uids, vids, clicks)

    print("\n" + "="*50)
    print(f"SONUÇ: Doküman Listesi İçin Model Tahminleri:")
    print("-" * 50)
    print(f"{'Pozisyon':<10} {'Doküman ID':<15} {'P(Inceleme)':<15} {'P(Çekicilik)':<15} {'Nihai Skor':<15}")
    print(f"{'----------':<10} {'---------------':<15} {'---------------':<15} {'---------------':<15} {'---------------':<15}")
    
    for i in range(10):
        doc_id = doc_ids[i]
        exam = exam_probs[0, i].item()
        attr = attr_scores[0, i].item()
        final = click_probabilities[0, i].item()
        print(f"{i+1:<10} {doc_id:<15} {exam:<15.4f} {attr:<15.4f} {final:<15.4f}")
    print("="*50)

if __name__ == "__main__":
    
    # argparse ile sadece değişecek olan girdileri (query_id, doc_ids) alalım.
    parser = argparse.ArgumentParser(description='GraphCM ile bir doküman listesini yeniden sıralama.')
    parser.add_argument('--query_id', type=int, required=True, help='Tahmin yapılacak sorgunun IDsi.')
    parser.add_argument('--doc_ids', type=str, required=True, help='JSON formatında 10 elemanlı doküman ID listesi. Örn: "[1,2,3,4,5,6,7,8,9,10]"')
    # Opsiyonel olarak hangi checkpoint'i yükleyeceğimizi de argüman olarak alalım
    parser.add_argument('--load_model', type=int, default=25825, help='Yüklenecek modelin adım numarası (checkpoint). Varsayılan: 5. epoch sonu.')

    script_args = parser.parse_args()

    # Modelin ihtiyaç duyduğu TÜM parametreleri içeren Namespace objesi.
    # Değerler, "emj_b128_lr0_001_full_cap" denemesine göre ayarlandı.
    model_args = argparse.Namespace(
        # --- TEMEL AYARLAR (Eğitim Komutundan Alındı) ---
        dataset='emj',
        model_dir='./outputs/models/emj_b128_lr0_001_full_cap',
        result_dir='./outputs/results/emj_b128_lr0_001_full_cap',
        summary_dir='./outputs/summary/emj_b128_lr0_001_full_cap',
        log_dir='./outputs/log/',
        algo='GraphCM',
        load_model=script_args.load_model, # Argümandan gelen değeri kullan
        
        # --- EĞİTİMDE KULLANILAN MİMARİ VE OPTIMIZER AYARLARI ---
        batch_size=128,
        optim='adam',
        learning_rate=0.001,
        embed_size=64,
        hidden_size=64,
        vtype_embed_size=8,
        click_embed_size=4,
        pos_embed_size=4,
        combine='mul',
        use_gnn=True,
        gnn_att_heads=2,
        weight_decay=1e-05,
        momentum=0.99,
        dropout_rate=0.5,
        gnn_neigh_sample=5,
        
        # --- Hata almamak için eklenen diğer zorunlu parametreler ---
        max_d_num=10,
        gnn_dropout=0,
        gnn_leaky_slope=0.2,
        gnn_concat=False,
        inter_neigh_sample=0,
        inter_leaky_slope=0.2,
        gpu_num=1,
        data_parallel=False,
        eval_freq=5165,
        check_point=5165,
        patience=5,
        lr_decay=0.5,
        train=False,
        valid=False,
        test=False,
        rank=False,
        num_iter=1,
        reg_relevance=1.0,
        use_pretrain_embed=False
    )
    
    model_args.query_id = script_args.query_id
    model_args.doc_ids = script_args.doc_ids

    rerank(model_args)