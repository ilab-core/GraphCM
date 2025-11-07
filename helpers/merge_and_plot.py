# -*- coding: utf-8 -*-
"""
Senaryo 1 ve Senaryo 2 için üretilen GraphCM tahminlerini,
orijinal IDBN tahminleriyle birleştirir ve 4'lü bir 
karşılaştırma grafiği çizer.

DÜZELTME: [Errno 2] No such file or directory hatasını önlemek için ve 
80M satır patlamasını engellemek için merge işleminden önce
'drop_duplicates' eklendi.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# --- Girdi Dosyaları ---
IDBN_CSV_FILE = 'compare/IDBN_click_probs.csv' # Hacer'in ana dosyası
S1_PREDS_FILE = 'compare/GraphCM_S1_Rank1_Predictions.csv' # Script 1'in çıktısı
S2_PREDS_FILE = 'compare/GraphCM_S2_Contextual_Predictions.csv' # Script 2'nin çıktısı

# --- Çıktı Dosyaları ---
FINAL_MERGED_CSV = 'compare/Final_All_Models_Comparison_UNIQUE.csv' # _UNIQUE eklendi
FINAL_PLOT_IMAGE = 'compare/Final_All_Models_Comparison_Plot_UNIQUE.png' # _UNIQUE eklendi

def main():
    print("--- Birleştirme ve Grafikle Görüntüleme Script'i (Düzeltilmiş) ---")

    try:
        # 1. IDBN ana verisini (S1 ile birleşik) yükle
        df_s1_with_idbn = pd.read_csv(S1_PREDS_FILE)
        
        # 2. Bizim S2 verimizi yükle
        df_s2 = pd.read_csv(S2_PREDS_FILE)
    except FileNotFoundError as e:
        print(f"HATA: Girdi dosyası bulunamadı: {e}")
        print("Lütfen 'run_scenario1_rank1.py' ve 'run_scenario2_contextual.py' script'lerini çalıştırdığınızdan emin olun.")
        return

    print(f"-> Yüklenen S1 verisi (IDBN + GraphCM S1): {len(df_s1_with_idbn)} satır")
    print(f"-> Yüklenen S2 verisi (GraphCM S2): {len(df_s2)} satır")

    # --- HATA DÜZELTMESİ (80 Milyon satır sorununu çözme) ---
    # Her iki DataFrame'de de (quid, uid, rank) bazında tekrarlanan satırları at.
    # Sadece benzersiz (unique) üçlüleri tut.
    print("-> Birleştirmeden önce tekrarlanan (quid, uid, rank) satırları kaldırılıyor...")
    
    key_cols = ['quid', 'uid', 'rank']
    
    df_s1_with_idbn.drop_duplicates(subset=key_cols, inplace=True)
    df_s2.drop_duplicates(subset=key_cols, inplace=True)
    
    print(f"   S1 verisi benzersiz (unique) satır sayısı: {len(df_s1_with_idbn)}")
    print(f"   S2 verisi benzersiz (unique) satır sayısı: {len(df_s2)}")
    # --- DÜZELTME SONU ---

    # --- FİNAL BİRLEŞTİRME ---
    print("-> S1 ve S2 tahminleri birleştiriliyor...")
    df_final = pd.merge(
        df_s1_with_idbn, 
        df_s2, 
        on=key_cols,
        how='inner' # Sadece iki sette de ortak olan satırları al
    )

    print(f"-> Toplam {len(df_final)} adet benzersiz eşleşen satır bulundu.")
    df_final.to_csv(FINAL_MERGED_CSV, index=False)
    print(f"-> Birleştirilmiş CSV kaydedildi: {FINAL_MERGED_CSV}")

    # --- GRAFİK ÇİZİMİ ---
    print("-> Karşılaştırma grafiği oluşturuluyor...")
    
    # Sütun Adlarını Okunaklı Hale Getir
    rename_map = {
        'full_click_prob': 'IDBN (S1: Rank=1)',
        'conditional_click_prob': 'IDBN (S2: Bağlamsal)',
        'graphcm_s1_rank1': 'GraphCM (S1: Rank=1)',
        'graphcm_s2_contextual': 'GraphCM (S2: Bağlamsal)'
    }
    df_final.rename(columns=rename_map, inplace=True)
    prob_columns = rename_map.values()
    
    df_long = df_final[prob_columns].melt(var_name='Model', value_name='Tahmin Edilen Olasılık (0-1)')
    
    plt.figure(figsize=(16, 9))
    
    sns.histplot(
        data=df_long,
        x='Tahmin Edilen Olasılık (0-1)',
        hue='Model',
        element='step',
        stat='density', 
        common_norm=False, 
        kde=True, 
        bins=100 
    )
    
    plt.title('Tıklama Modeli Senaryo Karşılaştırması (Test Seti)', fontsize=18)
    plt.xlabel('Tahmin Edilen Tıklama Olasılığı (0-1)', fontsize=14)
    plt.ylabel('Yoğunluk (Density)', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)    
    plt.savefig(FINAL_PLOT_IMAGE, dpi=150)
    print(f"-> Başarılı! Grafik şu dosyaya kaydedildi: {FINAL_PLOT_IMAGE}")

if __name__ == "__main__":
    main()