# -*- coding: utf-8 -*-

"""
Eğitilmiş GraphCM modelini kullanarak tek bir (sorgu, doküman) çifti için
tıklama olasılığı tahmini yapan script.

Örnek Komut:
python helpers/predict.py --query_id 5 --doc_id 120
"""

# 1. Gerekli Kütüphaneler ve Path Düzeltmesi
import torch
import argparse
import sys
import os

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
    # Model, tek bir ID yerine bir "batch" veri bekler.
    # Biz de tekil girdimizi 1'lik bir batch haline getiriyoruz.
    # Model 10'luk doküman listelerine göre eğitildiği için, girdimizi 0'larla dolduruyoruz.
    
    query_id = args.query_id
    doc_id = args.doc_id
    
    # Girdileri PyTorch tensörlerine çeviriyoruz.
    qids = torch.LongTensor([[query_id]]).to(device)
    uids = torch.LongTensor([[doc_id] + [0] * 9]).to(device) # İlk doküman bizimki, gerisi boş.
    vids = torch.LongTensor([[0] * 10]).to(device) # vtype için boş veri
    clicks = torch.LongTensor([[0] * 10]).to(device) # clicks için boş veri

    # 4. Tahmin Yapma ve Sonucu Gösterme
    # ----------------------------------------------------
    with torch.no_grad(): # Gradyan hesaplamasını kapatarak süreci hızlandırıyoruz.
        # Modeli çalıştırıp tıklama olasılıklarını alıyoruz.
        click_probabilities = model.model(qids, uids, vids, clicks)
    
    # Çıktı, [batch_size, sequence_length] boyutunda bir tensördür (bizim için [1, 10]).
    # Bizim ilgilendiğimiz ilk dokümanın olasılığını alıyoruz.
    prediction = click_probabilities[0, 0].item()

    print("\n" + "="*40)
    print(f"SONUÇ: Modelin bu dokümana tıklanma olasılığı tahmini:")
    print(f">>> {prediction:.4f} <<<")
    print("="*40)


# 5. Script'i Çalıştırma Bloğu
# ----------------------------------------------------
if __name__ == "__main__":
    
    # argparse ile sadece değişecek olan girdileri (query_id, doc_id) alalım.
    parser = argparse.ArgumentParser(description='GraphCM ile tekil tahmin yapma scripti.')
    parser.add_argument('--query_id', type=int, required=True, help='Tahmin yapılacak sorgunun IDsi.')
    parser.add_argument('--doc_id', type=int, required=True, help='Tahmin yapılacak dokümanın IDsi.')
    
    # Komut satırından gelen query_id ve doc_id'yi al.
    script_args = parser.parse_args()

    # Modelin geri kalan tüm ayarlarını içeren bir Namespace objesi oluşturalım.
    model_args = argparse.Namespace(
        dataset='emj',
        model_dir='./outputs/models/',
        result_dir='./outputs/results/',
        summary_dir='./outputs/summary/',
        log_dir='./outputs/log/',
        algo='GraphCM',
        load_model=8120,
        use_gnn=True,
        combine='mul',
        embed_size=32,
        hidden_size=64,
        max_d_num=10,
        pos_embed_size=4,
        click_embed_size=4,
        vtype_embed_size=8,
        gnn_neigh_sample=5,
        gnn_att_heads=2,
        gnn_dropout=0,
        gnn_leaky_slope=0.2,
        gnn_concat=False,
        inter_neigh_sample=0,
        inter_leaky_slope=0.2,
        optim='adadelta',
        learning_rate=0.01,
        weight_decay=1e-05,
        momentum=0.99,
        dropout_rate=0.5,
        gpu_num=0,
        vtype_size=1,
        eval_freq=99999,
        check_point=8120,
        patience=5,
        lr_decay=0.5,
        train=False,
        valid=False,
        test=False,
        rank=False,
        num_iter=1,
        reg_relevance=1.0,
        use_pretrain_embed=False,
        data_parallel=False
    )
    
    # Script'ten gelen argümanları model ayarlarına ekleyelim.
    model_args.query_id = script_args.query_id
    model_args.doc_id = script_args.doc_id

    # Ana tahmin fonksiyonunu çalıştır.
    predict(model_args)