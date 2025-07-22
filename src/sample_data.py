import os
import random
import shutil
from tqdm import tqdm


SOURCE_TRAIN_DIR = '/scratch/chaijy_root/chaijy2/janeding/imagenet/train'
SUBSET_DIR = ('/scratch/chaijy_root/chaijy2/janeding/imagenet/'
              'imagenet_train_subset_5_percent')
SAMPLE_RATIO = 0.05
RANDOM_SEED = 42


def create_stratified_subset():
    """
    Create a stratified subset of the ImageNet training set.
    """
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Construct the path for the subset's train directory
    subset_train_dir = SUBSET_DIR

    # If the target directory already exists, delete it to avoid confusion
    if os.path.exists(SUBSET_DIR):
        raise FileExistsError(f'Target directory {SUBSET_DIR} already exists.')

    print(f'Creating new subset directory: {subset_train_dir}')
    os.makedirs(subset_train_dir)

    # Get all class directories (e.g., 'n01440764', 'n01443537'...)
    class_dirs = [d for d in os.listdir(SOURCE_TRAIN_DIR)
                  if os.path.isdir(os.path.join(SOURCE_TRAIN_DIR, d))]
    assert len(class_dirs) == 1000, ('Expected 1000 classes, '
                                     f'got {len(class_dirs)}')

    print(f'Found {len(class_dirs)} classes. Starting stratified sampling...')

    num_sampled_classes = 0
    num_sampled_images = 0
    for class_name in tqdm(class_dirs, desc='Processing classes'):
        source_class_path = os.path.join(SOURCE_TRAIN_DIR, class_name)
        target_class_path = os.path.join(subset_train_dir, class_name)

        os.makedirs(target_class_path)

        # Get all image file names in the class
        images = [f for f in os.listdir(source_class_path)
                  if f.lower().endswith('.jpeg')]

        # Calculate the number of images to sample
        num_to_sample = int(len(images) * SAMPLE_RATIO)
        # Ensure at least one sample per class
        if num_to_sample == 0 and len(images) > 0:
            num_to_sample = 1

        # Randomly sample images from the list
        sampled_images = random.sample(images, num_to_sample)
        assert len(sampled_images) == num_to_sample, (
            f'Expected {num_to_sample} images, '
            f'got {len(sampled_images)}'
        )
        num_sampled_classes += 1

        # Copy the sampled images to the new subset directory
        for image_name in sampled_images:
            source_image_path = os.path.join(source_class_path, image_name)
            target_image_path = os.path.join(target_class_path, image_name)
            shutil.copyfile(source_image_path, target_image_path)
            num_sampled_images += 1

    print('\nStratified sampling completed!')
    print(f'New subset created at: {SUBSET_DIR}')
    print(f'Sampled classes: {num_sampled_classes}')
    print(f'Sampled images: {num_sampled_images}')


if __name__ == '__main__':
    create_stratified_subset()
