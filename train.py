import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from models.digit_classifier import DigitClassifier, TrainingReport
from tqdm import tqdm

def train_model(train_loader, test_loader, epochs=30, device="cuda"):
    model = DigitClassifier().to(device)
    
    # Training parameters
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=3e-3, epochs=epochs, 
                          steps_per_epoch=len(train_loader))
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize training report
    report = TrainingReport()
    report.set_training_params({
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": "3e-4 to 3e-3 (OneCycleLR)",
        "weight_decay": 0.01,
        "batch_size": train_loader.batch_size,
        "device": str(device)
    })
    
    report.set_dataset_info({
        "train_samples": len(train_loader.dataset),
        "test_samples": len(test_loader.dataset)
    })
    
    # Training loop
    best_acc = 0
    all_preds = []
    all_labels = []
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training phase
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", 
                         leave=False, unit="batch")
        for inputs, targets in batch_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*predicted.eq(targets).sum().item()/targets.size(0):.1f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
                if epoch == epochs - 1:  # Last epoch
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(targets.cpu().numpy())
        
        # Calculate metrics
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # Update report metrics
        report.update_metrics(
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc
        )
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_acc': f'{100.*train_acc:.1f}%',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_acc': f'{100.*val_acc:.1f}%'
        })
        
        # Save model if validation accuracy improves
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            tqdm.write(f"\n→ New best model saved with validation accuracy: {100.*val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    
    # Generate confusion matrix
    report.set_confusion_matrix(all_labels, all_preds)
    
    # Generate final report
    report_dir = report.generate_report()
    print(f"\nTraining complete! Report saved to: {report_dir}")
    print(f"Best validation accuracy: {100.*best_acc:.2f}%")
    
    return model, report 