import numpy as np
import torch

def train_network(net, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = np.inf
    best_model_state = None

    train_rmse_history = []
    val_rmse_history = []

    for epoch in range(num_epochs):
        net.train()
        train_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = net(xb).squeeze()
            loss = torch.sqrt(criterion(preds, yb))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        net.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = net(xb).squeeze()
                loss = torch.sqrt(criterion(preds, yb))
                val_losses.append(loss.item())

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = np.mean(val_losses)

        train_rmse_history.append(mean_train_loss)
        val_rmse_history.append(mean_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train RMSE: {mean_train_loss:.4f} | Val RMSE: {mean_val_loss:.4f}")

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = net.state_dict()

    return best_model_state, train_rmse_history, val_rmse_history
