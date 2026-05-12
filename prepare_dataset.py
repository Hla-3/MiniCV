import os
import csv
import random

# ============================================================
# Dataset Path Configuration
# Change this path if your dataset folder is named differently
# ============================================================
DATASET_PATH = "Dataset"   # Main dataset folder
OUTPUT_CSV   = "dataset.csv"

# Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)  # For reproducibility

# ============================================================
# Step 1: Read all images and create a list
# ============================================================
all_data = []  # Will contain (path, label)

classes = sorted(os.listdir(DATASET_PATH))
print(f"Classes Found: {classes}\n")

for class_name in classes:
    class_folder = os.path.join(DATASET_PATH, class_name)
    
    if not os.path.isdir(class_folder):
        continue
    
    # List images with specific extensions
    images = [f for f in os.listdir(class_folder)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in images:
        img_path = os.path.join(DATASET_PATH, class_name, img_name)
        all_data.append((img_path, class_name))
    
    print(f"  {class_name}: {len(images)} images")

print(f"\nTotal Images: {len(all_data)}")

# ============================================================
# Step 2: Stratified Split
# (Splitting per class to maintain a balanced distribution)
# ============================================================
train_data, val_data, test_data = [], [], []

for class_name in classes:
    # Filter images belonging to the current class
    class_images = [(p, l) for p, l in all_data if l == class_name]
    random.shuffle(class_images)
    
    n = len(class_images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)
    
    train_data += class_images[:n_train]
    val_data   += class_images[n_train:n_train + n_val]
    test_data  += class_images[n_train + n_val:]

# ============================================================
# Step 3: Save Data to CSV Files
# ============================================================
def save_csv(data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])  # header
        writer.writerows(data)
    print(f" Saved: {filename} ({len(data)} images)")

save_csv(train_data, "train.csv")
save_csv(val_data,   "val.csv")
save_csv(test_data,  "test.csv")
save_csv(all_data,   "dataset.csv")  # Full dataset file

# ============================================================
# Step 4: Print Summary Report
# ============================================================
print("\n========== Split Summary ==========")
print(f"Train : {len(train_data)} images")
print(f"Val   : {len(val_data)}  images")
print(f"Test  : {len(test_data)} images")
print("====================================")
print("\nCSV Format Example:")
print("image_path                          , label")
print("seg_test/buildings/img001.jpg       , buildings")
print("seg_test/forest/img001.jpg          , forest")
print("...")