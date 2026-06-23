import os
import shutil
import random
from pathlib import Path

# ==========================
# CONFIGURACIÓN
# ==========================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_DIR = BASE_DIR / "animals-dataset"
DESTINATION_DIR = BASE_DIR / "Animals_Dataset_Splitted"

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

random.seed(42)

# ==========================
# CREAR CARPETAS
# ==========================

for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(DESTINATION_DIR, split), exist_ok=True)

# ==========================
# RECORRER CADA ESPECIE
# ==========================

for species in sorted(os.listdir(SOURCE_DIR)):

    species_path = os.path.join(SOURCE_DIR, species)

    if not os.path.isdir(species_path):
        continue

    print(f"\nProcesando {species}")

    # La carpeta de imágenes es directamente la de la especie
    image_folder = species_path

    # Obtener imágenes
    images = [
        img for img in os.listdir(image_folder)
        if img.lower().endswith(IMAGE_EXTENSIONS)
    ]

    if not images:
        print(f"No se encontraron imágenes en {species}.")
        continue

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    train_images = images[:train_end]
    valid_images = images[train_end:valid_end]
    test_images = images[valid_end:]

    splits = {
        "train": train_images,
        "validation": valid_images,
        "test": test_images
    }

    for split_name, split_images in splits.items():

        dest_folder = os.path.join(
            DESTINATION_DIR,
            split_name,
            species
        )

        os.makedirs(dest_folder, exist_ok=True)

        for image in split_images:

            shutil.copy2(
                os.path.join(image_folder, image),
                os.path.join(dest_folder, image)
            )

    print(f"Total: {total}")
    print(f" Train: {len(train_images)}")
    print(f" Validation: {len(valid_images)}")
    print(f" Test: {len(test_images)}")

print("\n===================================")
print("Dataset creado correctamente.")
print("===================================")