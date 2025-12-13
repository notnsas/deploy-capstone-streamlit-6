import os

def print_tree(directory, prefix=''):
    """
    Fungsi rekursif untuk mencetak struktur folder dengan garis pohon.
    """
    # Cek apakah folder bisa diakses
    try:
        items = os.listdir(directory)
    except PermissionError:
        print(f"{prefix}└── 🚫 [Akses Ditolak]")
        return
    except FileNotFoundError:
        print(f"{prefix}└── ❌ [Folder Tidak Ditemukan]")
        return

    # Urutkan: Folder dulu, baru File (biar rapi seperti di Windows Explorer)
    # logic sort: (apakah file?, nama file) -> False (folder) muncul sebelum True (file)
    items.sort(key=lambda x: (not os.path.isdir(os.path.join(directory, x)), x.lower()))

    total_items = len(items)
    
    for i, item in enumerate(items):
        path = os.path.join(directory, item)
        is_last = (i == total_items - 1)
        
        # Tentukan simbol percabangan
        if is_last:
            connector = "└── "
            new_prefix = prefix + "    "
        else:
            connector = "├── "
            new_prefix = prefix + "│   "

        # Cek apakah item adalah Folder atau File
        if os.path.isdir(path):
            print(f"{prefix}{connector}📂 {item}")
            # PANGGIL LAGI fungsi ini untuk masuk ke dalam folder (Rekursif)
            print_tree(path, new_prefix)
        else:
            print(f"{prefix}{connector}📄 {item}")

# --- KONFIGURASI ---
# GANTI PATH DI BAWAH INI SESUAI FOLDER KAMU
folder_target = r"C:\Kuliah\Sains Data Sem 5\Data Science Capstone Project\MyCapstoneApp\SPOTIFY_ABSA_CAPSTONE_MODELS_FINAL_20251207_1600"

print("="*60)
print(f"📦 ROOT: {os.path.basename(folder_target)}")
print("="*60)

if os.path.exists(folder_target):
    print_tree(folder_target)
else:
    print(f"Error: Path '{folder_target}' tidak ditemukan.")
    
print("="*60)