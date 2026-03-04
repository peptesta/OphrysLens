# Script per trasformare le immagini ad una dimensione accettabile per il classificatore durante la fase di training

from pathlib import Path
from PIL import Image

def resize_smart(image_path, output_path):
    with Image.open(image_path) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
            
        width, height = img.size
        
        # Orientamento intelligente
        if height > width:
            target_size = (256, 512)
        else:
            target_size = (512, 256)
        
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        paste_x = (target_size[0] - img.size[0]) // 2
        paste_y = (target_size[1] - img.size[1]) // 2
        new_img.paste(img, (paste_x, paste_y))
        
        new_img.save(output_path, "JPEG", quality=95)

def main():
    script_dir = Path(__file__).resolve().parent

    base_path = script_dir
    input_base = base_path / "processed_results"
    output_base = base_path / "resized_results"
    
    # Estensioni supportate
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}
    
    folders = ["immagini_internet", "immagini_locali"]
    
    for folder_name in folders:
        source_root = input_base / folder_name
        dest_root = output_base / folder_name
        
        if not source_root.exists():
            print(f"⚠️ Cartella sorgente non trovata: {source_root}")
            continue

        print(f"🚀 Analisi ricorsiva di: {folder_name}...")
        
        count = 0
        # rglob("*") cerca in tutte le sottocartelle ricorsivamente
        for img_path in source_root.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                try:
                    # Manteniamo la struttura delle sottocartelle anche nella destinazione
                    relative_path = img_path.relative_to(source_root)
                    final_dest_path = dest_root / relative_path
                    
                    # Crea la sottocartella di destinazione se non esiste
                    final_dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    resize_smart(img_path, final_dest_path)
                    count += 1
                except Exception as e:
                    print(f"  ❌ Errore su {img_path.name}: {e}")
        
        print(f"  ✅ Finito! Elaborate {count} immagini per {folder_name}")

if __name__ == "__main__":
    main()