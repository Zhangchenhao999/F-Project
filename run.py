import argparse
from pathlib import Path
import torch
from models.digit_classifier import DigitClassifier
from train import train_model
from test import evaluate_model
from dataloader import create_combined_dataloaders

def main(args):
    # Optimized device selection for Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f'Using device: {device}')
    
    # Enable memory efficient mode for MPS
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.7)  # Prevent memory overflow
        
    # Setup directories
    base_dir = Path('data')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Load data
    member_dirs = [base_dir / f'dataset_member_{i}' for i in range(1, 4)]
    image_dirs = [base_dir / f'resized_images_member_{i}' for i in range(1, 4)]
    
    train_loader, test_loader = create_combined_dataloaders(
        member_dirs=member_dirs,
        image_dirs=image_dirs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        train_ratio=args.train_ratio
    )
    
    if args.mode == 'train':
        # Create model using DigitClassifier
        model = DigitClassifier().to(device)
        
        # Train model with updated train_model function
        model, report = train_model(
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            device=device,
        )
        
    elif args.mode == 'test':
        # Load model
        model = DigitClassifier().to(device)
        model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
        
        # Evaluate
        report, cm = evaluate_model(
            model, test_loader, device=device, save_dir=save_dir
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test digit classifier')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='train or test the model')
    parser.add_argument('--img-size', type=int, default=64,
                        help='image size for training')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size for training')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='ratio of data to use for training')
    parser.add_argument('--save-dir', type=str, default='models',
                        help='directory to save model and results')
    
    args = parser.parse_args()
    main(args) 