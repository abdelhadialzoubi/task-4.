import tensorflow as tf
import os
import pathlib
import shutil
from matplotlib import pyplot as plt

# Configuration
dataset_name = "facades"
dataset_url = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

# 1. Setup paths
download_dir = pathlib.Path('/root/.keras/datasets')
extract_dir = download_dir / f"{dataset_name}_extracted"
final_dir = download_dir / dataset_name

# 2. Clean up any previous attempts
if extract_dir.exists():
    shutil.rmtree(extract_dir)
if final_dir.exists():
    shutil.rmtree(final_dir)

# 3. Download the dataset
print("Downloading dataset...")
try:
    tf.keras.utils.get_file(
        fname=f"{dataset_name}.tar.gz",
        origin=dataset_url,
        extract=False,
        cache_dir=download_dir
    )
except Exception as e:
    print(f"Download failed: {e}")
    raise

# 4. Manual extraction
archive_path = download_dir / f"{dataset_name}.tar.gz"
print(f"\nExtracting {archive_path}...")
try:
    import tarfile
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
    print("Extraction successful")
    
    # 5. Find the actual dataset folder (it might be nested)
    extracted_folder = None
    for root, dirs, files in os.walk(extract_dir):
        if 'train' in dirs and ('test' in dirs or 'val' in dirs):
            extracted_folder = pathlib.Path(root)
            break
    
    if not extracted_folder:
        extracted_folder = extract_dir
    
    # 6. Move to final location
    shutil.move(str(extracted_folder), str(final_dir))
    print(f"Dataset moved to {final_dir}")
    
except Exception as e:
    print(f"Extraction failed: {e}")
    raise
finally:
    # Clean up
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    if archive_path.exists():
        archive_path.unlink()

# 7. Verify the dataset
print("\nFinal dataset structure:")
print(list(final_dir.iterdir()))

# 8. Load a sample image
train_dir = final_dir / 'train'
if not train_dir.exists():
    raise FileNotFoundError(f"Train directory not found in {final_dir}")

sample_image = next(train_dir.glob('*.jpg')) or next(train_dir.glob('*.png'))
if not sample_image:
    raise FileNotFoundError(f"No images found in {train_dir}")

print(f"\nLoading sample image: {sample_image}")
img = tf.io.read_file(str(sample_image))
img = tf.io.decode_jpeg(img)
print(f"Image shape: {img.shape}")

plt.figure()
plt.imshow(img)
plt.title("Sample Image")
plt.axis('off')
plt.show()