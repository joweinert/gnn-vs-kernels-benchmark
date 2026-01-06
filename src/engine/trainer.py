import torch
import time
import copy
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

class GNNTrainer:
    def __init__(self, model, optimizer, criterion, device=None, patience=20):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.patience = patience
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            
            # forward
            out = self.model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)
            
            # backward
            loss.backward()
            self.optimizer.step()
            
            # stats
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
            
        return total_loss / total, correct / total

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        total = 0
        
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data.x, data.edge_index, data.batch)
                loss = self.criterion(out, data.y)
                
                total_loss += loss.item() * data.num_graphs
                total += data.num_graphs
                
                # gets probabilities (softmax) for AUC
                probs = torch.softmax(out, dim=1)
                
                # stores for metric calculation
                all_preds.append(out.argmax(dim=1).cpu().numpy())
                all_labels.append(data.y.cpu().numpy())
                
                # if binary classification, stores prob of class 1
                if probs.shape[1] == 2:
                    all_probs.append(probs[:, 1].cpu().numpy())
                else:
                    all_probs.append(probs.cpu().numpy())

        # batch concat
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        
        # metrics
        acc = (all_preds == all_labels).mean()
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) == 2 else 0.0
        except ValueError:
            auc = 0.0

        return total_loss / total, acc, f1, auc

    def fit(self, train_loader, val_loader, epochs=50, verbose=False, restore_best=True):
        """
        restore_best: If True, restores the model weights that achieved best validation accuracy.
                      Set to False when 'val_loader' is actually the Test set.
        """
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
        
        start_time = time.perf_counter()
        
        epoch_iterator = tqdm(range(epochs), desc="Training GNN", unit="epoch")
        for epoch in epoch_iterator:
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1, val_auc = self.evaluate(val_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['val_auc'].append(val_auc)
            
            # updates tqdm description
            epoch_iterator.set_postfix({
                'T-Loss': f"{train_loss:.3f}", 
                'V-Acc': f"{val_acc:.3f}",
                'V-F1': f"{val_f1:.3f}"
            })
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # early stopping check
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if restore_best:
                    best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if restore_best and patience_counter >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break
                
        train_time = time.perf_counter() - start_time
        
        # load best model only if requested
        if restore_best and best_model_state:
            self.model.load_state_dict(best_model_state)
            
        return best_val_acc, history, train_time