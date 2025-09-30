import os
import random
import shutil

# --- CONFIGURATION ---
# 1. DEFINE YOUR SOURCE FOLDERS
#    Put all your original images here.
IMAGES_SOURCE_DIR = "dataset/allimage"
#    Put all your original .txt label files here.
LABELS_SOURCE_DIR = "dataset/labels-traffic"

# 2. DEFINE YOUR DESTINATION FOLDER
OUTPUT_DIR = "dataset"

# 3. SET YOUR SPLIT RATIO
#    0.8 means 80% for training, 20% for validation.
SPLIT_RATIO = 0.8

# --- SCRIPT ---
print("🚀 Starting dataset split...")

# Create the required output directories if they don't exist
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

# Get a list of all image files and check if the folder is empty
try:
    images = [f for f in os.listdir(IMAGES_SOURCE_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
    if not images:
        raise FileNotFoundError(f"No images found in '{IMAGES_SOURCE_DIR}'")
    random.shuffle(images)
except FileNotFoundError:
    print(f"❌ ERROR: Source images folder not found at '{IMAGES_SOURCE_DIR}'. Please check the path.")
    exit()

# Print feedback on what was found
print(f"✅ Found {len(images)} total images.")

# Calculate the split point
split_index = int(len(images) * SPLIT_RATIO)
train_images = images[:split_index]
val_images = images[split_index:]

print(f"   -> Assigning {len(train_images)} images to the training set.")
print(f"   -> Assigning {len(val_images)} images to the validation set.")

# Define a function to copy the files
def copy_files(image_list, split_name):
    copied_count = 0
    for img_filename in image_list:
        base_filename = os.path.splitext(img_filename)[0]
        label_filename = f"{base_filename}.txt"

        # Source paths
        src_image_path = os.path.join(IMAGES_SOURCE_DIR, img_filename)
        src_label_path = os.path.join(LABELS_SOURCE_DIR, label_filename)

        # Destination paths
        dest_image_path = os.path.join(OUTPUT_DIR, "images", split_name, img_filename)
        dest_label_path = os.path.join(OUTPUT_DIR, "labels", split_name, label_filename)

        # Copy image
        shutil.copy(src_image_path, dest_image_path)

        # IMPORTANT: Only copy the label if it actually exists
        if os.path.exists(src_label_path):
            shutil.copy(src_label_path, dest_label_path)
        
        copied_count += 1
    print(f"   -> Copied {copied_count} files to '{split_name}' folders.")

# Execute the copy process
copy_files(train_images, "train")
copy_files(val_images, "val")

print("\n🎉 Splitting complete! Your dataset is ready for training.")