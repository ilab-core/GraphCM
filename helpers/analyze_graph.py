import torch
from torch_geometric.utils import degree
import numpy as np

def analyze_graph(file_path, graph_name):
    """
    Verilen bir .pth dosyasındaki grafı analiz eder ve istatistikleri yazdırır.
    """
    print(f"--- {graph_name} Analizi ---")
    
    try:
        # .pth dosyasını yükle
        edge_index = torch.load(file_path)
        
        # Eğer graf boşsa 
        if edge_index.shape[1] == 0:
            print(f"Graf boş veya hatalı: {file_path}")
            print("-" * 25 + "\n")
            return

        # Temel istatistikleri hesapla
        num_edges = edge_index.shape[1]
        
        # Düğüm sayısını en yüksek ID'den tahmin et
        # Not: Graf bağlantısız bileşenler içeriyorsa bu sayı gerçek düğüm sayısından az olabilir
        # ama derece hesaplaması için yeterlidir.
        num_nodes = edge_index.max().item() + 1
        
        # edge_index[0] kaynak düğümleri içerir
        node_degrees = degree(edge_index[0], num_nodes=num_nodes)
        
        # Derece istatistiklerini hesapla
        avg_degree = node_degrees.mean().item()
        median_degree = np.median(node_degrees.numpy())
        max_degree = node_degrees.max().item()
        
        print(f"Yüklenen dosya: {file_path}")
        print(f"Toplam Düğüm (Node) Sayısı: {num_nodes}")
        print(f"Toplam Kenar (Edge) Sayısı: {num_edges}")
        print("-" * 15)
        print(f"Ortalama Düğüm Derecesi: {avg_degree:.2f}")
        print(f"Medyan Düğüm Derecesi: {median_degree:.2f}")
        print(f"MAKSİMUM Düğüm Derecesi: {max_degree}")
        print("-" * 25 + "\n")

    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# --- Analizi Başlat ---

print("=" * 30)
print("     TAM VERİ SETİ (emj)")
print("=" * 30)
analyze_graph('data/emj/dgat_qid_edge_index.pth', 'Sorgu Grafı (Query Graph)')
analyze_graph('data/emj/dgat_uid_edge_index.pth', 'Doküman Grafı (Document Graph)')


print("=" * 30)
print("     %25 VERİ SETİ (25_percent)")
print("=" * 30)
analyze_graph('data/25_percent/dgat_qid_edge_index.pth', 'Sorgu Grafı (Query Graph)')
analyze_graph('data/25_percent/dgat_uid_edge_index.pth', 'Doküman Grafı (Document Graph)')