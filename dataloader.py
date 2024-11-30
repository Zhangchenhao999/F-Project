from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class DigitClassificationDataset(Dataset):
    def __init__(self, img_files, label_files, img_size=64, transform=None):
        """
        Dataset for digit classification using cropped digits from object detection dataset.
        
        Args:
            img_files: List of paths to image files
            label_files: List of paths to label files
            img_size: Size to resize cropped digits to (default: 64 for better resolution)
            transform: Optional transforms to apply
        """
        self.img_files = img_files
        self.label_files = label_files
        self.img_size = img_size
        self.transform = transform
        
        # Pre-process all crops and labels
        self.crops = []
        self.labels = []
        self._load_all_samples()

    def _load_all_samples(self):
        for img_path, label_path in zip(self.img_files, self.label_files):
            try:
                # Load directly as grayscale
                img = Image.open(img_path).convert('L')
                img_width, img_height = img.size
                
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        
                        # Convert YOLO coordinates to pixel coordinates
                        x1 = int((x_center - width/2) * img_width)
                        y1 = int((y_center - height/2) * img_height)
                        x2 = int((x_center + width/2) * img_width)
                        y2 = int((y_center + height/2) * img_height)
                        
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_width, x2), min(img_height, y2)
                        
                        # Crop and resize the digit
                        if x2 > x1 and y2 > y1:
                            crop = img.crop((x1, y1, x2, y2))
                            crop = crop.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
                            
                            # No need to convert to grayscale again
                            self.crops.append(crop)
                            self.labels.append(int(class_id))
                            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop = self.crops[idx]
        label = self.labels[idx]
        
        # Convert to tensor (will be single channel)
        crop = transforms.ToTensor()(crop)
        
        # Apply transforms if specified
        if self.transform:
            crop = self.transform(crop)
            
        return crop, label

def create_combined_dataloaders(
    member_dirs: list,  # List of member directories containing labels
    image_dirs: list = None,  # Optional list of image directories
    batch_size: int = 32,
    img_size: int = 64,
    train_ratio: float = 0.7,
    num_workers: int = 2,  # Reduced for M-series efficiency
    random_seed: int = 42,
    pin_memory: bool = True,
    persistent_workers: bool = True
):
    """
    Create optimized dataloaders for M-series Macs.
    
    Args:
        member_dirs: List of directories containing the label (.txt) files
        image_dirs: List of directories containing the image files (optional)
        batch_size: Batch size for training
        img_size: Size to resize cropped digits to (default: 64 for better resolution)
        train_ratio: Ratio of data to use for training
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
        pin_memory: Whether to use pin_memory for data loading
        persistent_workers: Whether to use persistent workers
    
    Returns:
        train_loader, test_loader: DataLoader objects for training and testing
    """
    # Set random seed
    torch.manual_seed(random_seed)
    
    # If image_dirs not provided, use member_dirs
    if image_dirs is None:
        image_dirs = member_dirs
    
    if len(image_dirs) != len(member_dirs):
        raise ValueError("Number of image directories must match number of member directories")
    
    # Collect valid pairs from all members
    all_valid_pairs = []
    
    for label_dir, image_dir in zip(member_dirs, image_dirs):
        label_path = Path(label_dir)
        image_path = Path(image_dir)
        
        print(f"\nProcessing dataset:")
        print(f"Labels: {label_path}")
        print(f"Images: {image_path}")
        
        # Get all label files
        label_files = sorted(list(label_path.glob('*.txt')))
        if not label_files:
            print(f"Warning: No .txt files found in {label_path}")
            continue
        
        print(f"Found {len(label_files)} label files")
        
        # Find corresponding image files
        member_valid_pairs = []
        for label_file in label_files:
            # Try both .jpg and .png
            img_jpg = image_path / f"{label_file.stem}.jpg"
            img_png = image_path / f"{label_file.stem}.png"
            
            if img_jpg.exists():
                member_valid_pairs.append((img_jpg, label_file))
            elif img_png.exists():
                member_valid_pairs.append((img_png, label_file))
            else:
                print(f"Warning: No image found for {label_file.name}")
        
        print(f"Found {len(member_valid_pairs)} valid pairs")
        all_valid_pairs.extend(member_valid_pairs)
    
    if not all_valid_pairs:
        raise ValueError("No valid image-label pairs found in any directory!")
    
    print(f"\nTotal valid pairs across all datasets: {len(all_valid_pairs)}")
    
    # Split into train and test sets
    img_files, label_files = zip(*all_valid_pairs)
    
    X_train, X_test, y_train, y_test = train_test_split(
        img_files, label_files,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )
    
    # Enhanced transforms for grayscale images
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(15),
        # Add contrast and brightness adjustment
        transforms.RandomAdjustSharpness(sharpness_factor=2),
        transforms.RandomAutocontrast(),
        # Normalize to [-1, 1] range
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_transform = transforms.Compose([
        # Add contrast adjustment for test set too
        transforms.RandomAutocontrast(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Create datasets
    train_dataset = DigitClassificationDataset(X_train, y_train, img_size, train_transform)
    test_dataset = DigitClassificationDataset(X_test, y_test, img_size, test_transform)
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2,
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2
    )
    
    print(f"\nCombined dataset split complete:")
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    return train_loader, test_loader



