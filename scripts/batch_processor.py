import os
import requests
import time
from PIL import Image

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:5000/dbinference" 
INPUT_ROOT = "./input_images"      
OUTPUT_ROOT = "./processed_results" 
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
BATCH_SIZE = 10         
CONFIDENCE_THRESHOLD = 90  

TARGET_V = (256, 512) # Vertical
TARGET_H = (512, 256) # Horizontal

processing_summary = []

def center_on_bbox(img: Image.Image, bbox: list, output_size=None):
    """
    Centra l'immagine sulla bounding box.

    Args:
        img (PIL.Image): immagine originale.
        bbox (list): bounding box [x1, y1, x2, y2].
        output_size (tuple, optional): dimensione (width, height) finale dell'immagine.
                                       Se None, mantiene la dimensione del ritaglio centrato.
    
    Returns:
        PIL.Image: immagine centrata sulla bounding box.
    """
    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) // 2
    bbox_center_y = (y1 + y2) // 2

    img_width, img_height = img.size

    # Determiniamo la dimensione del ritaglio
    if output_size:
        crop_width, crop_height = output_size
    else:
        # Default: larghezza/altezza della bounding box * 2 per un po' di padding
        crop_width = (x2 - x1) * 2
        crop_height = (y2 - y1) * 2

    # Calcoliamo i limiti del ritaglio centrati sulla bounding box
    left = max(0, bbox_center_x - crop_width // 2)
    top = max(0, bbox_center_y - crop_height // 2)
    right = min(img_width, bbox_center_x + crop_width // 2)
    bottom = min(img_height, bbox_center_y + crop_height // 2)

    cropped_img = img.crop((left, top, right, bottom))

    # Se specificata output_size, ridimensioniamo
    if output_size:
        cropped_img = cropped_img.resize(output_size, Image.Resampling.LANCZOS)

    return cropped_img


def process_folder(folder_path, folder_name):
    all_entries = [e for e in os.listdir(folder_path) 
                   if os.path.splitext(e)[1].lower() in ALLOWED_EXTENSIONS]
    
    total_images = len(all_entries)
    if total_images == 0:
        print(f"Skipping folder '{folder_name}': no images found.")
        return

    est_seconds_total = total_images * 1.25
    est_min, est_sec = divmod(int(est_seconds_total), 60)

    folder_start_time = time.time()
    total_batches = (total_images + BATCH_SIZE - 1) // BATCH_SIZE
    
    print("-" * 50)
    print(f"STARTING FOLDER: {folder_name}")
    print(f"Total images: {total_images} | Batches: {total_batches}")
    print(f"Estimated time: {est_min}m {est_sec}s")
    print("-" * 50)

    total_crops_saved = 0

    for i in range(0, total_images, BATCH_SIZE):
        current_batch_idx = (i // BATCH_SIZE) + 1
        batch_files_names = all_entries[i : i + BATCH_SIZE]
        num_images_in_batch = len(batch_files_names)
        files_to_upload = []
        
        for img_name in batch_files_names:
            full_path = os.path.join(folder_path, img_name)
            files_to_upload.append(('images', (img_name, open(full_path, 'rb'), 'image/jpeg')))

        print(f"Sending batch {current_batch_idx}/{total_batches} ({num_images_in_batch} images)...")

        try:
            response = requests.post(API_URL, files=files_to_upload)
            response.raise_for_status()
            data = response.json()

            current_out_dir = os.path.join(OUTPUT_ROOT, folder_name)
            os.makedirs(current_out_dir, exist_ok=True)

            saved_in_this_batch = 0
            for j, img_name in enumerate(batch_files_names):
                if j >= len(data['bounding_box']): break
                
                boxes = data['bounding_box'][j]
                scores = data['scores'][j]
                
                if not boxes: continue

                full_img_path = os.path.join(folder_path, img_name)
                with Image.open(full_img_path).convert("RGB") as img:
                    # Sequential counter for crops within the SAME image
                    image_crop_counter = 1 
                    
                    for idx, (box, score) in enumerate(zip(boxes, scores)):
                        if (score * 100) >= CONFIDENCE_THRESHOLD:
                            # 1️⃣ Centriamo l'immagine sulla bounding box
                            centered_crop = center_on_bbox(img, box)

                            # 2️⃣ Applichiamo il ridimensionamento con padding dinamico
                            #final_crop = resize_with_dynamic_padding(centered_crop, TARGET_V, TARGET_H)
                            final_crop = centered_crop

                            base_name = os.path.splitext(img_name)[0]

                            # Use originalname_1.jpg, originalname_2.jpg, etc.
                            save_name = f"{base_name}_{image_crop_counter}.jpg"
                            save_path = os.path.join(current_out_dir, save_name)

                            final_crop.save(save_path, "JPEG", quality=95)

                            image_crop_counter += 1
                            saved_in_this_batch += 1

            
            total_crops_saved += saved_in_this_batch
            print(f"Batch {current_batch_idx} completed. Crops: {saved_in_this_batch}")

        except Exception as e:
            print(f"Error in batch {current_batch_idx}: {e}")
        finally:
            for _, f_tuple in files_to_upload: f_tuple[1].close()
        
        time.sleep(0.5)

    folder_duration = time.time() - folder_start_time
    processing_summary.append({
        'folder': folder_name,
        'images': total_images,
        'crops': total_crops_saved,
        'duration': folder_duration
    })

    print(f"FINISHED FOLDER: {folder_name} | Actual: {int(folder_duration // 60)}m {int(folder_duration % 60)}s")
    print("-" * 50)

def main():
    if not os.path.exists(INPUT_ROOT):
        os.makedirs(INPUT_ROOT)
        print(f"Folder '{INPUT_ROOT}' created. Add images and restart.")
        return

    global_start_time = time.time()
    print(f"INITIALIZING PROCESS (Threshold: {CONFIDENCE_THRESHOLD}%)")
    
    subfolders = [f for f in os.scandir(INPUT_ROOT) if f.is_dir()]
    
    if not subfolders:
        process_folder(INPUT_ROOT, "root_images")
    else:
        for folder in subfolders:
            process_folder(folder.path, folder.name)

    total_duration = time.time() - global_start_time
    
    print("\n" + "=" * 65)
    print(f"{'FOLDER NAME':<25} | {'IMAGES':<8} | {'CROPS':<8} | {'TIME':<10}")
    print("-" * 65)
    
    total_imgs = total_crops = 0
    for item in processing_summary:
        m, s = divmod(int(item['duration']), 60)
        print(f"{item['folder'][:25]:<25} | {item['images']:<8} | {item['crops']:<8} | {m}m {s}s")
        total_imgs += item['images']
        total_crops += item['crops']

    print("-" * 65)
    t_m, t_s = divmod(int(total_duration), 60)
    print(f"{'TOTAL':<25} | {total_imgs:<8} | {total_crops:<8} | {t_m}m {t_s}s")
    print("=" * 65)

if __name__ == "__main__":
    main()


def resize_with_dynamic_padding(img, size_v, size_h):

    ###########
    return img
    ###########
    w, h = img.size
    original_aspect = w / h
    
    ratio_v = size_v[0] / size_v[1]
    ratio_h = size_h[0] / size_h[1]
    
    if abs(original_aspect - ratio_v) < abs(original_aspect - ratio_h):
        target_size = size_v
    else:
        target_size = size_h

    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    new_img = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_size[0] - img.size[0]) // 2
    paste_y = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    
    return new_img