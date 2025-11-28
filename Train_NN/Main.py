import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from GiveData import give_data
from NeuralNetwork import Net
from TrainNeuralNet import train_network

import os

# Absolute path where you want to save the model
save_dir = r"path where to save"
save_filename = "Network_full.pth"
save_path = os.path.join(save_dir, save_filename)

filename = r""  # DataSet32Patch3 excel file + path
sheet = 7 
df = pd.read_excel(filename, sheet_name=sheet)


#save_filename = "Network2.pth"
#filename = r"\\nl-filer1\users$\maaike\Desktop\python\Train_NN\training_data-total.txt"
#df = pd.read_csv(filename, sep=";", decimal=",")

train_loader, val_loader, test_loader, num_in = give_data(df)

net = Net(num_in=num_in, num_neurons=16, num_layers=2, dropout=0.0)

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

num_epochs = 5000
best_model_state, train_rmse_history, val_rmse_history = train_network(net, train_loader, val_loader, criterion, optimizer, num_epochs)

net.load_state_dict(best_model_state)  # load best model
net.eval()
test_losses = []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = net(xb).squeeze()
        loss = torch.sqrt(criterion(preds, yb))  # RMSE
        test_losses.append(loss.item())

print(f"Final Test RMSE: {np.mean(test_losses):.4f}")

torch.save(net, save_path)
print(f"✅ Model saved at: {save_path}")
print("Model and normalization parameters saved.")

plt.figure(figsize=(8,5))
plt.plot(train_rmse_history, label="Train RMSE", color='tab:blue')
plt.plot(val_rmse_history, label="Validation RMSE", color='tab:red', marker='o')
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title("Training vs Validation RMSE")
plt.grid(True)
plt.legend()
plt.show()
