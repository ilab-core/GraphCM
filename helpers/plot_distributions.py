# -*- coding: utf-8 -*-
"""
'run_comparison_prediction.py' script'inin ürettiği CSV dosyasını okur
ve 3 farklı tıklama modelinin (IDBN-full, IDBN-cond, GraphCM)
tahmin dağılımlarını gösteren bir grafik (histogram/KDE) çizer.

(Düzeltilmiş Versiyon: Lejant hatası giderildi ve isimler netleştirildi)
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 'run_comparison_prediction.py' script'inin ürettiği dosya
INPUT_FILE = 'compare/IDBN_click_probs_with_GraphCM.csv'
OUTPUT_IMAGE = 'compare/distribution_comparison_v2.png' # Çıktı adını değiştirelim

def plot_distributions(df):
    print(f"-> '{INPUT_FILE}' dosyasındaki verilerle grafik oluşturuluyor...")
    
    # --- İYİLEŞTİRME 1: Sütun Adlarını Okunaklı Hale Getir ---
    rename_map = {
        'full_click_prob': 'Hacer (Genel CTR)',
        'conditional_click_prob': 'Hacer (Bağlamsal)',
        'graphcm_prob': 'GraphCM '
    }
    df.rename(columns=rename_map, inplace=True)
    prob_columns = rename_map.values() # ['Hacer (Genel CTR)', 'Hacer (Bağlamsal)', 'GraphCM (Bizim Model)']
    # --- İYİLEŞTİRME 1 SONU ---

    # Veriyi 'uzun' (long) formata çevir (seaborn için daha kolay)
    df_long = df[prob_columns].melt(var_name='Model', value_name='Tahmin Edilen Olasılık (0-1)')
    
    # Grafik boyutunu ayarla
    plt.figure(figsize=(14, 8))
    
    # Histogram ve KDE (Kernel Density Estimation) grafiğini birlikte çiz
    sns.histplot(
        data=df_long,
        x='Tahmin Edilen Olasılık (0-1)',
        hue='Model', # Bu, 'Hacer (Genel CTR)', 'Hacer (Bağlamsal)' vb. isimlerini alacak
        element='step',
        stat='density', 
        common_norm=False, 
        kde=True, 
        bins=100 
    )
    
    plt.title('Tıklama Modeli Tahmin Dağılımları Karşılaştırması (Test Seti)', fontsize=16)
    plt.xlabel('Tahmin Edilen Tıklama Olasılığı (0-1)', fontsize=12)
    plt.ylabel('Yoğunluk (Density)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- HATA DÜZELTMESİ ---
    # Seaborn 'hue' kullandığında zaten lejantı otomatik oluşturur.
    # Bu satır, otomatik oluşan lejantı eziyordu, bu yüzden kaldırıldı.
    # plt.legend(title='Model') 
    # --- HATA DÜZELTMESİ SONU ---

    # Grafiği dosyaya kaydet
    try:
        plt.savefig(OUTPUT_IMAGE)
        print(f"-> Başarılı! Düzeltilmiş grafik şu dosyaya kaydedildi: {OUTPUT_IMAGE}")
    except Exception as e:
        print(f"HATA: Grafik kaydedilemedi: {e}")

def main():
    print("Dağılım Grafiği Script'i Başlatıldı.")
    
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"HATA: Girdi dosyası bulunamadı: {INPUT_FILE}")
        print("Lütfen önce 'helpers/run_comparison_prediction.py' script'ini çalıştırdığınızdan emin olun.")
        return
        
    if 'graphcm_prob' not in df.columns:
        print(f"HATA: '{INPUT_FILE}' dosyasında 'graphcm_prob' sütunu bulunamadı.")
        print("Lütfen 'helpers/run_comparison_prediction.py' script'ini tekrar çalıştırın.")
        return
        
    plot_distributions(df)

if __name__ == "__main__":
    main()