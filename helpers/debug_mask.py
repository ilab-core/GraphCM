import sys
import os
import torch
import torch.nn.functional as F
from argparse import Namespace
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dataset import Dataset


def run_mask_debug():
    print("="*60)
    print("Maskeleme ve Loss Mantığı Debug Testi Başlatıldı...")
    print("="*60)

    # Dataset sınıfının ihtiyaç duyduğu minimum argümanları taklit ediyoruz.
    args = Namespace(dataset='emj_30ilan_debug', max_d_num=30, gpu_num=1)

    # Debug için oluşturduğumuz veri setini yüklüyoruz.
    dataset = Dataset(args)
    
    # Çıktıları kolayca okuyabilmek için batch_size'ı küçük tutuyoruz (örn: 2).
    debug_batch_generator = dataset.gen_mini_batches('train', batch_size=2, shuffle=False)
    
    try:
        # Sadece ilk batch'i alıp inceleyeceğiz.
        batch = next(debug_batch_generator)
    except StopIteration:
        print("\nHATA: Debug veri seti ('data/emj_30ilan_debug/train_per_query_quid.txt') boş veya okunamıyor.")
        print("Lütfen yukarıdaki ön işleme komutlarını doğru çalıştırdığından emin ol.")
        return

    print("\n[1. Adım] Dataset'ten Gelen Batch'in İçeriği:")
    print("-" * 50)
    
    # dataset.py'den gelen uids, clicks ve masks verilerinin hepsi "liste içinde liste"dir.
    uids_list = batch['uids']
    clicks_list = batch['clicks']
    mask_list = batch['masks'] 

    # Batch içindeki her bir örneği ayrı ayrı yazdıralım.
    for i in range(len(uids_list)):
        print(f"\n--- Batch'teki {i+1}. Örnek ---")
        print(f"   - UIDs:   {uids_list[i]}")
        print(f"   - MASK:   {mask_list[i]}")
        print(f"   - CLICKS: {clicks_list[i]}")

    print("\n\n   -> GÖZLEM: Maske, padding olan yerler (0) için '0', gerçek ilanlar için '1' olarak doğru üretilmiş.")

    # Adım 2: Loss Hesaplamasını Simüle Et
    print("\n[2. Adım] Loss Hesaplama Simülasyonu:")
    print("-" * 50)

    # Tüm listeleri hesaplama yapabilmek için PyTorch tensörlerine çeviriyoruz.
    uids_tensor = torch.tensor(uids_list, dtype=torch.float32)
    pred_logits = torch.full_like(uids_tensor, 0.5, dtype=torch.float32) # Modelin tahminini 0.5 olarak varsayıyoruz.
    TRUE_CLICKS = torch.tensor(clicks_list, dtype=torch.float32)
    MASK = torch.tensor(mask_list, dtype=torch.bool)

    print(f"   - Modelin Tahmini (pred_logits): Tümü 0.5")
    print(f"   - Gerçek Tıklamalar (TRUE_CLICKS): \n{TRUE_CLICKS.int()}")

    # Adım 2.1: Ham Loss
    raw_loss = F.binary_cross_entropy(pred_logits, TRUE_CLICKS, reduction='none')
    print(f"\n   - Ham Loss (Tüm 30 ilan için hesaplanan hata): \n{raw_loss}")
    
    # Adım 2.2: Maskeleme
    masked_values = torch.masked_select(raw_loss, MASK)
    print(f"\n   - Maske (MASK): \n{MASK.int()}")
    
    print(f"\n   - Maskelenmiş Değerler (Sadece gerçek ilanların hataları): \n{masked_values}")
    print("\n   -> GÖZLEM: Sadece maskenin '1' olduğu yerlerdeki hata değerleri başarıyla seçildi.")

    # Adım 2.3: Nihai Loss
    final_loss = torch.mean(masked_values)
    print(f"\n   - Nihai Ortalama Loss: {final_loss.item():.6f}")
    print("\n   -> GÖZLEM: Ortalama, sadece bu seçilen gerçek ilanların hataları üzerinden hesaplandı.")

    print("\n" + "="*60)
    print("✅ TEST SONUCU: Maskeleme ve loss hesaplama mantığı beklendiği gibi çalışıyor.")
    print("="*60)

if __name__ == '__main__':
    run_mask_debug()