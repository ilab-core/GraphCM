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
    parser = argparse.ArgumentParser(description='GraphCM ile bir doküman listesini yeniden sıralama.')
    parser.add_argument('--query_id', type=int, required=True, help='Tahmin yapılacak sorgunun IDsi.')
    parser.add_argument('--doc_ids', type=str, required=True, help='JSON formatında 10 elemanlı doküman ID listesi. Örn: "[1,2,3,4,5,6,7,8,9,10]"')
    
    script_args = parser.parse_args()

    # Modelin ihtiyaç duyduğu TÜM parametreleri içeren Namespace objesi.
    # Değerler, son başarılı eğitime göre ayarlandı.
    model_args = argparse.Namespace(
        dataset='25_percent',
        model_dir='./outputs/models/emj_b256_final',
        result_dir='./outputs/results/emj_b256_final',
        summary_dir='./outputs/summary/emj_b256_final',
        log_dir='./outputs/log/emj_b256_final',
        algo='GraphCM',
        load_model=7740,
        
        # --- SON BAŞARILI EĞİTİMDEKİ MİMARİ VE DİĞER AYARLAR ---
        batch_size=256,
        optim='adam',
        learning_rate=0.001,
        embed_size=32,
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
        max_d_num=10,
        gnn_dropout=0,
        gnn_leaky_slope=0.2,
        gnn_concat=False,
        inter_neigh_sample=0,
        inter_leaky_slope=0.2,
        gpu_num=1,
        data_parallel=False,
        eval_freq=999999,
        check_point=645,
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