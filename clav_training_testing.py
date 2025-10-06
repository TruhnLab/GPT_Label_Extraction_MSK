import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from tempfile import TemporaryDirectory
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import time
import copy
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, roc_curve
import pandas as pd
from pandas import ExcelWriter
import re
import json

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, phase='train', transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.phase = phase

        # Filter data based on phase and reset index
        self.data = self.data[self.data['Phase'] == self.phase].reset_index(drop=True)

        self.image_paths = self.data['Image_Path'].tolist()
        self.labels = self.data.iloc[:, 1:-1].values  # Exclude 'Image_Path' and 'Split' columns
        self.class_names = self.data.columns[1:-1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if isinstance(img_path, np.ndarray):
            img_path = img_path.item()  
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, labels, img_path

def main(num_epochs, batch_size):
    # Define file paths and save filenames (using resnet50)
    model_save_path = f'Output/clavicle_batch_size_{batch_size}_resnet50.pt'
    threshold_save_path = f'Output/clavicle_batch_size_{batch_size}_resnet50.json'
    csv_file = 'Output/clavicle.csv'  # Updated to use output from data preparation
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

    
    full_dataset = MultiLabelDataset(csv_file, transform=data_transforms['train'])

    # Create datasets for each phase
    train_dataset = MultiLabelDataset(csv_file, phase='train', transform=data_transforms['train'])
    val_dataset = MultiLabelDataset(csv_file, phase='val', transform=data_transforms['val'])
    test_dataset = MultiLabelDataset(csv_file, phase='test', transform=data_transforms['test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set number of classes and create ResNet50 model
    num_classes = len(full_dataset.data.columns) - 2  # Exclude 'Image_Path' and 'Split'
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    class_names = list(full_dataset.class_names)

    # Compute class weights for imbalance handling
    all_labels = pd.read_csv(csv_file).iloc[:, 1:-1].values
    class_weights = []
    for class_name in class_names:
        class_count = np.sum(all_labels[:, class_names.index(class_name)])
        class_weights.append(class_count)
    total_count = np.sum(class_weights)
    class_weights = [weight / total_count for weight in class_weights]

    def train_model(model, criterion, optimizer, scheduler, num_epochs, save_path, batch_size, class_weights):
        since = time.time()
        best_auc = 0.0

        # Use a temporary directory for intermediate checkpoint saving
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
            torch.save(model.state_dict(), best_model_params_path)

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()
                    else:
                        model.eval()

                    running_loss = 0.0
                    all_labels = []
                    all_probs = []

                    for inputs, labels, _ in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device).float()  # For BCEWithLogitsLoss

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            probs = torch.sigmoid(outputs).cpu().detach().numpy()
                            loss = criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        running_loss += loss.item() * inputs.size(0)
                        all_labels.extend(labels.cpu().numpy())
                        all_probs.extend(probs)

                    epoch_loss = running_loss / dataset_sizes[phase]
                    all_labels = np.array(all_labels)
                    all_probs = np.array(all_probs)
                    weighted_auc = 0.0
                    for i in range(num_classes):
                        class_auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                        weighted_auc += class_weights[i] * class_auc
                    epoch_auc = weighted_auc

                    print(f'{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}')

                    # Step the scheduler only during training
                    if phase == 'train':
                        scheduler.step()
                        epoch_train_loss = epoch_loss
                        epoch_train_auc = epoch_auc
                    else:
                        epoch_val_loss = epoch_loss
                        epoch_val_auc = epoch_auc

                    # Save the best model based on validation AUC
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

    model_ft = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs,
                           model_save_path, batch_size, class_weights)

    def find_optimal_thresholds(model, dataloader, device, class_names):
        model.eval()
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for image, labels, _ in dataloader:
                image = image.to(device)
                labels = labels.to(device).float()
                outputs = model(image)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        optimal_thresholds = []

        for i, class_name in enumerate(class_names):
            y_true = all_labels[:, i]
            y_probs = all_probs[:, i]
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)
            youden_index = tpr - fpr
            optimal_idx = np.argmax(youden_index)
            optimal_threshold = thresholds[optimal_idx]
            optimal_thresholds.append(float(optimal_threshold))

        return optimal_thresholds

    optimal_thresholds = find_optimal_thresholds(model_ft, dataloaders['val'], device, class_names)
    with open(threshold_save_path, 'w') as f:
        json.dump(optimal_thresholds, f)

    def evaluate_model(optimal_thresholds, model, dataloaders, dataset_sizes, device, class_names, batch_size):
        model.eval()
        all_labels = []
        all_probs = []
        img_paths = []

        with torch.no_grad():
            for image, labels, img_path in dataloaders['test']:
                image = image.to(device)
                labels = labels.to(device).float()
                outputs = model(image)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs)
                img_paths.extend(img_path)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'roc_auc': [],
            'positive_labels': []
        }

        # Create a directory for ROC plots if it doesn't exist
        os.makedirs('elbow', exist_ok=True)

        for i, class_name in enumerate(class_names):
            y_true = all_labels[:, i]
            y_probs = all_probs[:, i]
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

            # Plot ROC Curve and save to file
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
            plt.scatter([closest_threshold_fpr], [closest_threshold_tpr],
                        color='red', label=f'Threshold {optimal_thresholds[i]:.2f}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {class_name}')
            plt.legend(loc='best')
            plot_filename = f'Output/roc_curve_{sanitized_class_name}.png'
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved ROC curve plot for {class_name} to {plot_filename}")

        results = pd.DataFrame({
            'Image_Path': img_paths,
            **{class_name: all_labels[:, i] for i, class_name in enumerate(class_names)},
            **{f'{class_name}_pred': (all_probs[:, i] >= optimal_thresholds[i]).astype(int)
               for i, class_name in enumerate(class_names)}
        })

       

        metrics_df = pd.DataFrame(metrics, index=class_names)

        # Save results and metrics to an Excel file
        excel_filename = f'Output/results_with_predictions_resnet50_bs{batch_size}.xlsx'
        os.makedirs(os.path.dirname(excel_filename), exist_ok=True)
        with ExcelWriter(excel_filename) as writer:
            results.to_excel(writer, sheet_name='Results', index=False)
            metrics_df.to_excel(writer, sheet_name='Metrics')

        print("Metrics and results saved.")

    evaluate_model(optimal_thresholds, model_ft, dataloaders, dataset_sizes, device, class_names, batch_size)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs for training')
    args = parser.parse_args()

    main(args.num_epochs, args.batch_size)
