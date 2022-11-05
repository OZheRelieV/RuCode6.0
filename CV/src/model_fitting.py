import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import f1_score as f1


def _fit_epoch(model, train_loader, optimizer, criterion=nn.CrossEntropyLoss()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_loss, count = 0.0, 0
    y_true, y_preds = [], []
  
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(outputs, 1)
        
        current_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)
        y_true += labels.cpu().numpy().tolist()
        y_preds += preds.cpu().numpy().tolist()
              
    train_loss = current_loss / count
    train_f1 = f1(y_true, y_preds, average="macro")
    
    return train_loss, train_f1


def _eval_epoch(model, val_loader, criterion=nn.CrossEntropyLoss()):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    
    current_loss, count  = 0.0, 0
    y_true, y_preds = [], []

    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            preds = torch.argmax(outputs, 1)

        current_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)
        y_true += labels.cpu().numpy().tolist()
        y_preds += preds.cpu().numpy().tolist()
        
    val_loss = current_loss / count
    val_f1 = f1(y_true, y_preds, average="macro")
    
    return val_loss, val_f1


def train(train_loader, val_loader, model, epochs, validate=True):
    
    info = []
    with tqdm(desc="epoch", total=epochs) as pbar_outer:
        opt = torch.optim.Adamax(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(opt, 3, 0.5)

        for epoch in range(epochs):
            train_loss, train_f1 = _fit_epoch(model, train_loader, opt)
            if validate:
                val_loss, val_f1 = _eval_epoch(model, val_loader)
                info.append((train_loss, train_f1, val_loss, val_f1))
            else:
                info.append((train_loss, train_f1))
            scheduler.step()
            pbar_outer.update(1)
            if validate:
                tqdm.write(f"[{epoch}]: tr_loss: {round(train_loss, 5)} val_loss: {round(val_loss, 5)}\
                tr_f1: {round(train_f1, 5)} val_f1: {round(val_f1, 5)}")
            else:
                tqdm.write(f"[{epoch}]: tr_loss: {round(train_loss, 5)} tr_f1: {round(train_f1, 5)} ")
            
    return info


def predict(model, test_loader, inverse_target_mapping=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        logits = []
        for inputs in test_loader:
            inputs = inputs.to(device)
            model.eval()
            outputs = model(inputs).cpu()
            logits.append(outputs)
            
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    y_pred = np.argmax(probs, -1)
    
    return [inverse_target_mapping[i] for i in y_pred] if inverse_target_mapping else y_pred