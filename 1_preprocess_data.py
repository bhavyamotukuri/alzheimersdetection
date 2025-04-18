import os
import shutil
from sklearn.model_selection import train_test_split


original_dataset = "/Users/bhavya.motukuri.11/Desktop/ALZHEIMER/newdataset"
output_dir = "data_split" 
def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))
def split_dataset():
   
    categories = [d for d in os.listdir(original_dataset) if os.path.isdir(os.path.join(original_dataset, d))]

   
    for category in categories:
        category_path = os.path.join(original_dataset, category)

        images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        for split, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(output_dir, split, category)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                src = os.path.join(category_path, img)
                dst = os.path.join(split_dir, img)
                shutil.copy(src, dst)

    print("✅ Dataset split into train, val, and test.")


if __name__ == "__main__":
    split_dataset()

