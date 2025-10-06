import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
import re
from pandas import ExcelWriter
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, phase='train', transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.phase = phase
        
        # Filter data based on phase and reset index
        self.data = self.data[self.data['Phase'] == self.phase].reset_index(drop=True)
        
        self.ap_image_paths = self.data['AP_Image_Path'].tolist()
        self.lat_image_paths = self.data['LAT_Image_Path'].tolist()
        self.labels = self.data.iloc[:, 2:-1].values  # Exclude 'AP_Image_Path' and 'Phase' columns
        self.class_names = self.data.columns[2:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ap_img_path = self.ap_image_paths[idx]
        lat_img_path = self.lat_image_paths[idx]
        ap_image = Image.open(ap_img_path).convert('RGB')
        lat_image = Image.open(lat_img_path).convert('RGB')
        
        if self.transform:
            ap_image = self.transform(ap_image)
            lat_image = self.transform(lat_image)
        
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return ap_image, lat_image, labels, ap_img_path, lat_img_path

class DualResNet(nn.Module):
    def __init__(self, num_classes):
        super(DualResNet, self).__init__()
        # Only using ResNet50 for both branches
        self.ap_resnet = models.resnet50(pretrained=True)
        self.lat_resnet = models.resnet50(pretrained=True)
        
        num_ftrs = self.ap_resnet.fc.in_features
        # Remove the final fully connected layers by replacing with Identity
        self.ap_resnet.fc = nn.Identity()
        self.lat_resnet.fc = nn.Identity()
        
        # Concatenate features from both networks and project to the number of classes
        self.fc = nn.Linear(num_ftrs * 2, num_classes)
        
    def forward(self, ap_image, lat_image):
        ap_features = self.ap_resnet(ap_image)
        lat_features = self.lat_resnet(lat_image)
        combined_features = torch.cat((ap_features, lat_features), dim=1)
        output = self.fc(combined_features)
        return output

def main(num_epochs, batch_size):
    # Set file paths
    csv_file = 'Output/Elbow.csv'  # Updated to use output from data preparation
    
    model_save_path = f'Output/elbow_batch_size_{batch_size}_resnet50.pt'
    threshold_save_path = f'Output/elbow_batch_size_{batch_size}_resnet50_elbow.json'
    
    # Define image transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }
    
    # Load datasets
    train_dataset = MultiLabelDataset(csv_file, phase='train', transform=data_transforms['train'])
    val_dataset = MultiLabelDataset(csv_file, phase='val', transform=data_transforms['val'])
    test_dataset = MultiLabelDataset(csv_file, phase='test', transform=data_transforms['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Dataset Sizes:")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}
    class_names = list(train_dataset.class_names)
    num_classes = len(class_names)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Compute class weights (normalized to sum to 1)
    all_labels = pd.read_csv(csv_file).iloc[:, 2:-1].values
    class_weights = []
    for class_name in class_names:
        class_count = np.sum(all_labels[:, class_names.index(class_name)])
        class_weights.append(class_count)
    total_count = np.sum(class_weights)
    class_weights = [weight / total_count for weight in class_weights]
    
    model = DualResNet(num_classes)
    model = model.to(device)
    
    def train_model(model, criterion, optimizer, scheduler, num_epochs, save_path, class_weights):
        since = time.time()
        best_auc = 0.0
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                    
                running_loss = 0.0
                all_labels_epoch = []
                all_probs_epoch = []
                
                for ap_image, lat_image, labels, _, _ in dataloaders[phase]:
                    ap_image = ap_image.to(device)
                    lat_image = lat_image.to(device)
                    labels = labels.to(device).float()
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(ap_image, lat_image)
                        probs = torch.sigmoid(outputs).cpu().detach().numpy()
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    running_loss += loss.item() * ap_image.size(0)
                    all_labels_epoch.extend(labels.cpu().numpy())
                    all_probs_epoch.extend(probs)
                
                epoch_loss = running_loss / dataset_sizes[phase]
                all_labels_epoch = np.array(all_labels_epoch)
                all_probs_epoch = np.array(all_probs_epoch)
                weighted_auc = 0.0
                for i in range(num_classes):
                    class_auc = roc_auc_score(all_labels_epoch[:, i], all_probs_epoch[:, i])
                    weighted_auc += class_weights[i] * class_auc
                epoch_auc = weighted_auc
                
                print(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')
                
                if phase == 'train':
                    scheduler.step()
                if phase == 'val' and epoch_auc >= best_auc:
                    best_auc = epoch_auc
                    torch.save(model.state_dict(), save_path)
                    
            print()
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val AUC: {best_auc:.4f}')
        
        model.load_state_dict(torch.load(save_path))
        return model
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler,
                           num_epochs, model_save_path, class_weights)
    
    def find_optimal_thresholds(model, dataloader, device, class_names):
        model.eval()
        all_labels_thresh = []
        all_probs_thresh = []
        
        with torch.no_grad():
            for ap_image, lat_image, labels, _, _ in dataloader:
                ap_image = ap_image.to(device)
                lat_image = lat_image.to(device)
                labels = labels.to(device).float()
                outputs = model(ap_image, lat_image)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels_thresh.extend(labels.cpu().numpy())
                all_probs_thresh.extend(probs)
        
        all_labels_thresh = np.array(all_labels_thresh)
        all_probs_thresh = np.array(all_probs_thresh)
        optimal_thresholds = []
        for i, _ in enumerate(class_names):
            y_true = all_labels_thresh[:, i]
            y_probs = all_probs_thresh[:, i]
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(float(optimal_threshold))
        return optimal_thresholds
    
    optimal_thresholds = find_optimal_thresholds(model_ft, val_loader, device, class_names)
    with open(threshold_save_path, 'w') as f:
        json.dump(optimal_thresholds, f)
    
    def evaluate_model(optimal_thresholds, model, dataloaders, device, class_names, batch_size):
        model.eval()
        all_labels_eval = []
        all_probs_eval = []
        ap_img_paths = []
        lat_img_paths = []
        
        with torch.no_grad():
            for ap_image, lat_image, labels, ap_img_path, lat_img_path in dataloaders['test']:
                ap_image = ap_image.to(device)
                lat_image = lat_image.to(device)
                labels = labels.to(device).float()
                outputs = model(ap_image, lat_image)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels_eval.extend(labels.cpu().numpy())
                all_probs_eval.extend(probs)
                ap_img_paths.extend(ap_img_path)
                lat_img_paths.extend(lat_img_path)
                
        all_labels_eval = np.array(all_labels_eval)
        all_probs_eval = np.array(all_probs_eval)
        
        metrics = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'roc_auc': [],
            'positive_labels': []
        }
        
        for i, class_name in enumerate(class_names):
            y_true = all_labels_eval[:, i]
            y_probs = all_probs_eval[:, i]
            preds = (y_probs >= optimal_thresholds[i]).astype(int)
            
            acc = accuracy_score(y_true, preds)
            sens = recall_score(y_true, preds)
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            try:
                auc = roc_auc_score(y_true, y_probs)
            except ValueError:
                auc = np.nan
            
            metrics['accuracy'].append(acc)
            metrics['sensitivity'].append(sens)
            metrics['specificity'].append(spec)
            metrics['roc_auc'].append(auc)
            metrics['positive_labels'].append(np.sum(y_true))
            
            sanitized_class_name = re.sub(r'\W+', '_', class_name)
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            closest_idx = (np.abs(thresholds - optimal_thresholds[i])).argmin()
            closest_threshold_fpr = fpr[closest_idx]
            closest_threshold_tpr = tpr[closest_idx]
            
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            plt.scatter([closest_threshold_fpr], [closest_threshold_tpr],
                        color='red', label=f'Threshold {optimal_thresholds[i]:.2f}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {class_name}')
            plt.legend(loc='best')
            plot_filename = f'Output/roc_curve_resnet50_{batch_size}_{sanitized_class_name}.png'
            os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved ROC curve for {class_name} to {plot_filename}")
        
        results = pd.DataFrame({
            'AP_Image_Path': ap_img_paths,
            **{class_name: all_labels_eval[:, i] for i, class_name in enumerate(class_names)},
            **{f'{class_name}_pred': (all_probs_eval[:, i] >= optimal_thresholds[i]).astype(int)
               for i, class_name in enumerate(class_names)}
        })
        
        metrics_df = pd.DataFrame(metrics, index=class_names)
        excel_filename = f'Output/results_with_predictions_resnet50_bs{batch_size}.xlsx'
        os.makedirs(os.path.dirname(excel_filename), exist_ok=True)
        with ExcelWriter(excel_filename) as writer:
            results.to_excel(writer, sheet_name='Results', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics')
        
        print("Metrics and results saved to Excel.")
    
    evaluate_model(optimal_thresholds, model_ft, dataloaders, device, class_names, batch_size)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    args = parser.parse_args()
    
    main(args.num_epochs, args.batch_size)
