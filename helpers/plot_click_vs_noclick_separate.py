# -*- coding: utf-8 -*-
"""
'run_click_vs_noclick_analysis.py' script'inin ürettiği CSV'yi okur.
İsteğe uygun olarak, 'Gerçek Click' olan ilanların dağılımı ile
'Gerçek Click Olmayan' ilanların dağılımını
İKİ AYRI YAN YANA GRAFİKTE gösterir.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# --- Girdi ve Çıktı Dosyaları ---
INPUT_CSV_FILE = 'compare/click_vs_noclick_probs.csv'
OUTPUT_PLOT_IMAGE = 'compare/click_vs_noclick_separate_plot.png'

def main():
    print("--- Click vs. No-Click AYRI Dağılım Grafiği ---")

    try:
        df = pd.read_csv(INPUT_CSV_FILE)
    except FileNotFoundError:
        print(f"HATA: Girdi dosyası bulunamadı: {INPUT_CSV_FILE}")
        print("Lütfen önce 'helpers/run_click_vs_noclick_analysis.py' script'ini çalıştırdığınızdan emin olun.")
        return

    if not {'probability', 'is_click'}.issubset(df.columns):
        print("HATA: CSV dosyasında 'probability' veya 'is_click' sütunları eksik.")
        return
        
    print(f"-> {len(df)} satır veri yüklendi.")
    
    # --- İKİ AYRI GRAFİK İÇİN VERİYİ FİLTRELE ---
    
    # 'clicks' listesi (Gerçekte 1 olanlar)
    df_clicks = df[df['is_click'] == 1]
    
    # 'no_clicks' listesi (Gerçekte 0 olanlar)
    df_no_clicks = df[df['is_click'] == 0]
    
    print(f"   Gerçek Click Sayısı (clicks listesi): {len(df_clicks)}")
    print(f"   Gerçek Click Yok Sayısı (no_clicks listesi): {len(df_no_clicks)}")

    # --- GRAFİK ÇİZİMİ (1 satır, 2 sütun) ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True) # Y eksenini (Yoğunluk) paylaş
    
    fig.suptitle('Bağlamsal Senaryo', fontsize=20, y=1.03)

    # --- GRAFİK 1: Sadece Tıklananlar (clicks listesi) ---
    sns.histplot(
        data=df_clicks,
        x='probability',
        element='step',
        stat='density', 
        common_norm=False, 
        kde=True, 
        bins=100,
        color='green', # Tıklananları yeşil yapalım
        ax=axes[0] # Sol taraftaki grafiğe çiz
    )
    axes[0].set_title(f'Gerçekte TIKLANAN İlanların Tahmin Dağılımı\n(clicks listesi - {len(df_clicks)} adet)', fontsize=16)
    axes[0].set_xlabel('Modelin Tahmin Ettiği Olasılık (0-1)', fontsize=12)
    axes[0].set_ylabel('Yoğunluk (Density)', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # --- GRAFİK 2: Sadece Tıklanmayanlar (no_clicks listesi) ---
    sns.histplot(
        data=df_no_clicks,
        x='probability',
        element='step',
        stat='density', 
        common_norm=False, 
        kde=True, 
        bins=100,
        color='red', # Tıklanmayanları kırmızı yapalım
        ax=axes[1] # Sağ taraftaki grafiğe çiz
    )
    axes[1].set_title(f'Gerçekte TIKLANMAYAN İlanların Tahmin Dağılımı\n(no_clicks listesi - {len(df_no_clicks)} adet)', fontsize=16)
    axes[1].set_xlabel('Modelin Tahmin Ettiği Olasılık (0-1)', fontsize=12)
    axes[1].set_ylabel('') # Y ekseni paylaşıldığı için etiketi gizle
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout() # Grafikleri düzgünce sığdır
    
    try:
        plt.savefig(OUTPUT_PLOT_IMAGE, dpi=150, bbox_inches='tight')
        print(f"-> Başarılı! İki ayrı grafiği içeren dosya şuraya kaydedildi: {OUTPUT_PLOT_IMAGE}")
    except Exception as e:
        print(f"HATA: Grafik kaydedilemedi: {e}")

if __name__ == "__main__":
    main()