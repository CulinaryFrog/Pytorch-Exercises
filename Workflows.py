import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path
# Check PyTorch version
print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

##1
weight = 0.3
bias = 0.9
X_values = torch.arange(0, 1, .01).unsqueeze(1) # Originaly had 0-100, but later problem specifies learning rate of .01, so not optimal
Y_values = weight * X_values + bias
split_percent = 80
X_train = X_values[:split_percent]
Y_train = Y_values[:split_percent]
X_test = X_values[split_percent:]
Y_test = Y_values[split_percent:]

def plot_sol1(train_data, train_labels, test_data, test_labels, predict_data = None, predict_labels = None):
    plt.figure(figsize = (10, 7))
    plt.scatter(train_data, train_labels, c = "g", s = 4, label = "Training")
    plt.scatter(test_data, test_labels, c = "y", s = 4, label = "Test")
    if (predict_data is not None and predict_labels is not None): # Will come up in solution 4
        plt.scatter(predict_data, predict_labels, c = "b", s = 4, label = "Prediction")

plot_sol1(X_train, Y_train, X_test, Y_test)
plt.show()


##2

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        self.weights = nn.Parameter(data = torch.rand(1, requires_grad = True, dtype=torch.float))
        self.bias = nn.Parameter(data = torch.rand(1, requires_grad = True, dtype=torch.float))

    def forward(self, x):
        return self.weights * x + self.bias

sol2_model = LinearRegressionModel()
print(sol2_model.state_dict())
sol2_model.to(device) # Send model to the relevant device


##3

sol3_loss_fn = nn.L1Loss()
sol3_optimnizer = torch.optim.SGD(params = sol2_model.parameters(), lr = 0.01)
training_epochs = 300
test_epochs = 20
X_train = X_train.to(device) # Need to send these tensors to correct device, otherwise error occurs. Also a note that tensors on GPU cannot be plotted with matplotlib.
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)

for i in range(1,training_epochs+1):
    sol2_model.train()
    Y_pred = sol2_model(X_train)
    loss = sol3_loss_fn(Y_pred, Y_train)
    sol3_optimnizer.zero_grad()
    loss.backward()
    sol3_optimnizer.step()

    if (i % test_epochs == 0): # Every 20 epochs test
        sol2_model.eval()
        with torch.inference_mode():
            
            Y_preds = sol2_model(X_test)
            test_loss = sol3_loss_fn(Y_preds, Y_test)
            print(f"Epoch: {i} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss}")


##4

sol2_model.eval()
with torch.inference_mode(): # Create new predictions
    sol4_Y_preds = sol2_model(X_test)
print(sol4_Y_preds)



plot_sol1(X_train.cpu(), Y_train.cpu(), X_test.cpu(), Y_test.cpu(), X_test.cpu(), sol4_Y_preds.cpu())  # Need to send these tensors back to CPU
plt.show()


##5

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)

MODEL_NAME = "sol4_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
torch.save(obj = sol2_model.state_dict(), f = MODEL_SAVE_PATH)

# Reload and check
sol5_model = LinearRegressionModel()
sol5_model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
sol5_model.to(device)

with torch.inference_mode(): # Create new predictions
    sol5_Y_preds = sol5_model(X_test)
print(sol5_Y_preds)


print(sol5_Y_preds == sol4_Y_preds)