import os
import argparse
import ast # <--- Güvenli okuma için bu kütüphaneyi ekledik

def load_literal_dict(file_path):
    """
    Python sözlüğü olarak yazılmış bir metin dosyasını güvenli bir şekilde okur.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        try:
            # ast.literal_eval, metni güvenli bir şekilde Python nesnesine çevirir
            dictionary = ast.literal_eval(content)
            return dictionary
        except Exception as e:
            print(f"HATA: {file_path} dosyası okunurken bir sorun oluştu: {e}")
            return None

def find_text_by_id(data_dir, query_id_to_find, doc_id_to_find):
    """
    Verilen ID'lere karşılık gelen sorgu metnini ve doküman URL'ini bulur.
    """
    query_dict_path = os.path.join(data_dir, 'query_qid.dict')
    doc_dict_path = os.path.join(data_dir, 'url_uid.dict')

    print(f"Sözlükler okunuyor: {data_dir}")

    # Sözlükleri yeni özel fonksiyonumuzla yükle
    query_dict = load_literal_dict(query_dict_path)
    doc_dict = load_literal_dict(doc_dict_path)

    if query_dict is None or doc_dict is None:
        return

    # Sözlükler "metin": id formatında. Biz id: "metin" aradığımız için tersine çevirelim.
    # Bu sefer ID'ler (v) integer, metinler (k) integer veya string olabilir.
    query_id_to_text = {v: k for k, v in query_dict.items()}
    doc_id_to_text = {v: k for k, v in doc_dict.items()}
    
    print("-" * 30)

    # Aradığımız Query ID'yi bul ve yazdır
    if query_id_to_find is not None:
        # ID'ler artık integer olduğu için str() kullanmıyoruz
        query_text = query_id_to_text.get(query_id_to_find, "BULUNAMADI")
        print(f"Sorgu ID: {query_id_to_find} -> Metin: {query_text}")

    # Aradığımız Doc ID'yi bul ve yazdır
    if doc_id_to_find is not None:
        doc_text = doc_id_to_text.get(doc_id_to_find, "BULUNAMADI")
        print(f"Doküman ID: {doc_id_to_find} -> URL: {doc_text}")
    
    print("-" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query ve Document ID\'lerine karşılık gelen metinleri bulur.')
    parser.add_argument('--query_id', type=int, help='Metni bulunacak sorgunun IDsi.')
    parser.add_argument('--doc_id', type=int, help='URL\'i bulunacak dokümanın IDsi.')
    args = parser.parse_args()

    if not args.query_id and not args.doc_id:
        print("Lütfen en az bir --query_id veya --doc_id parametresi girin.")
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_directory = os.path.join(project_root, 'data', 'emj')
        find_text_by_id(data_directory, args.query_id, args.doc_id)