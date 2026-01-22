import os
import time
import yaml
import argparse
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    defaults = {
        'root_dir': 'ocr_dataset',
        'blank_char': '-',
        'seed': 0,
        'val_size': 0.1,
        'test_size': 0.1,
        'shuffle': True,
        'img_height': 100,
        'img_width': 420,
        'train_batch_size': 64,
        'test_batch_size': 128,
        'hidden_size': 256,
        'n_layers': 3,
        'dropout': 0.2,
        'unfreeze_layers': 3,
        'epochs': 120,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'max_grad_norm': 1,
        'patience': 5,
        'save_model_path': 'model/ocr_crnn.pt',
        'early_stop_path': 'early_stop_model.pt',
    }
    
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
    
    return config

def encode(label, char_to_idx, max_label_len):
    label = label.lower()
    encoded = [char_to_idx[c] for c in label if c in char_to_idx]
    encoded_labels = torch.tensor(encoded, dtype=torch.long)
    label_len = len(encoded_labels)
    lengths = torch.tensor(label_len, dtype=torch.long)
    padded_labels = F.pad(encoded_labels, (0, max_label_len - label_len), value=0)
    return padded_labels, lengths

class STRDataset(Dataset):
    def __init__(self, X, y, char_to_idx, max_label_len, label_encoder=None, transform=None):
        self.transform = transform
        self.img_paths = X
        self.labels = y
        self.char_to_idx = char_to_idx
        self.max_label_len = max_label_len
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        if self.label_encoder:
            encoded_label, label_len = self.label_encoder(label, self.char_to_idx, self.max_label_len)
        return img, encoded_label, label_len

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout=0.3, unfreeze_layers=3):
        super(CRNN, self).__init__()

        backbone = timm.create_model("resnet34", in_chans=1, pretrained=False)
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for parameter in self.backbone[-unfreeze_layers:].parameters():
            parameter.requires_grad = True

        self.mapSeq = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout))

        self.gru = nn.GRU(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.out = nn.Sequential(nn.Linear(hidden_size * 2, vocab_size), nn.LogSoftmax(dim=2))

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.gru(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)
        return x

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels, labels_len in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            outputs = model(inputs)
            logits_lens = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels, logits_lens, labels_len)
            losses.append(loss.item())

    return sum(losses) / len(losses)

def fit(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, max_grad_norm, patience, early_stop_path):
    train_losses = []
    val_losses = []
    patience_counter = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        start = time.time()
        batch_train_losses = []

        model.train()
        for inputs, labels, labels_len in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            logits_lens = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, labels.cpu(), logits_lens.cpu(), labels_len.cpu())
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        print(f"EPOCH {epoch + 1}:\tTrain loss: {train_loss:.4f}\tVal loss: {val_loss:.4f}\t\tTime: {time.time() - start:.2f} seconds")

        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            torch.save(model.state_dict(), early_stop_path)
            print(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses

def export_to_onnx(model, save_path, img_height, img_width):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 1, img_height, img_width).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {1: 'batch_size'}
        },
        verbose=False
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='train_ocr_config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    root_dir = config['root_dir']
    img_paths = []
    labels = []
    
    with open(os.path.join(root_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
        for lb in f:
            parts = lb.strip().split('\t')
            if len(parts) == 2:
                img_paths.append(parts[0])
                labels.append(parts[1])
    
    print(f'Loaded {len(labels)} images')
    
    letters = [char.split('.')[0].lower() for char in labels]
    letters = ''.join(letters)
    letters = sorted(list(set(list(letters))))
    chars = ''.join(letters) + config['blank_char']
    vocab_size = len(chars)
    
    print(f'Vocab: {chars}')
    print(f'Vocab size: {vocab_size}')
    
    max_label_len = max([len(label) for label in labels])
    print(f'Max label length: {max_label_len}')
    
    char_to_index = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((config['img_height'], config['img_width'])),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.GaussianBlur(3),
            transforms.RandomAffine(degrees=1, shear=1),
            transforms.RandomPerspective(distortion_scale=0.3, p=0.5, interpolation=3),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
        "val": transforms.Compose([
            transforms.Resize((config['img_height'], config['img_width'])),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]),
    }
    
    X_train, x_val, y_train, y_val = train_test_split(
        img_paths, labels,
        test_size=config['val_size'],
        random_state=config['seed'],
        shuffle=config['shuffle']
    )
    
    x_train, x_test, y_train, y_test = train_test_split(
        X_train, y_train,
        test_size=config['test_size'],
        random_state=config['seed'],
        shuffle=config['shuffle']
    )
    
    train_dataset = STRDataset(x_train, y_train, char_to_idx=char_to_index, max_label_len=max_label_len, label_encoder=encode, transform=data_transforms["train"])
    val_dataset = STRDataset(x_val, y_val, char_to_idx=char_to_index, max_label_len=max_label_len, label_encoder=encode, transform=data_transforms["val"])
    
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['test_batch_size'], shuffle=False)
    
    model = CRNN(
        vocab_size=vocab_size,
        hidden_size=config['hidden_size'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        unfreeze_layers=config['unfreeze_layers'],
    ).to(device)
    
    criterion = nn.CTCLoss(blank=char_to_index[config['blank_char']], zero_infinity=True, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'], eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    train_losses, val_losses = fit(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, config['epochs'], config['max_grad_norm'], 
        config['patience'], config['early_stop_path']
    )
    
    torch.save(model.state_dict(), config['save_model_path'])
    print(f"Model checkpoint: {config['save_model_path']}")
    
    onnx_path = config['save_model_path'].replace('.pt', '.onnx')
    export_to_onnx(model, onnx_path, config['img_height'], config['img_width'])
    print(f"ONNX model: {onnx_path}")

if __name__ == '__main__':
    main()