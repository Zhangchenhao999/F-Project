import torch
from pathlib import Path
from models.digit_classifier import DigitClassifier
from dataloader import create_combined_dataloaders
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device='cuda', save_dir='models'):
    save_dir = Path(save_dir)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, digits=4)
    print("\nClassification Report:")
    print(report)
    
    # Save report
    with open(save_dir / 'test_report.txt', 'w') as f:
        f.write(report)
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_dir / 'confusion_matrix.png')
    plt.close()
    
    return report, cm

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    base_dir = 'Tensor Observer'
    member_dirs = [f'{base_dir}/dataset_member_{i}' for i in range(1, 4)]
    image_dirs = [f'{base_dir}/resized_images_member_{i}' for i in range(1, 4)]
    
    _, test_loader = create_combined_dataloaders(
        member_dirs=member_dirs,
        image_dirs=image_dirs,
        batch_size=32,
        img_size=28,
        train_ratio=0.7
    )
    
    # Load model
    model = DigitClassifier().to(device)
    checkpoint = torch.load('models/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    report, cm = evaluate_model(model, test_loader, device=device) 