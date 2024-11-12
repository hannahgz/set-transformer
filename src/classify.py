import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

    def forward(self, x):
        return self.fc(x)  # No activation, as we will use CrossEntropyLoss which applies softmax internally

def prepare_data(X, y, test_size=0.2, random_state=42):
    """Splits the data into train and test sets and converts to torch tensors."""
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.long)
    # X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=100, batch_size=32, patience=10):
    """Trains the model on the training data."""

    wandb.init(
        project="classify-card-attribute",
        config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "patience": patience,
        }
    )

    model.train()  # Set the model to training mode

    counter = 0
    best_loss = 1e9
    for epoch in range(num_epochs):
        if counter > patience:
            print("Exceeded patience: ", patience) 
            break
        permutation = torch.randperm(X_train.size(0))
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
    
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss every epoch
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": loss,
            }
        )
        
        if loss < best_loss:
            counter = 0
            best_loss = loss

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_num": epoch,
                "best_loss": best_loss,
            }
            torch.save(checkpoint, f'{PATH_PREFIX}/classify/test_model.pt')
        else:
            counter += 1


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test data."""
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

def run_classify(X, y, input_dim=16, output_dim=12, num_epochs=100, batch_size=32, lr=0.01):
    """Main function to run the model training and evaluation."""
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Train the model
    train_model(model, X_train, y_train, criterion, optimizer, num_epochs, batch_size)

    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)

    wandb.log({"test_accuracy": accuracy})
    wandb.finish()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")



# Example usage:
# X = [...]  # Your input data as a list of vectors (num_samples, 16)
# y = [...]  # Your target labels (num_samples,) where values are 0 to 11
# main(X, y)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# # Assuming your dataset looks like this:
# # X - List of input vectors, each of shape (16,)
# # y - Corresponding class labels from 0 to 11

# # Example data (replace this with your actual data)
# X = [...]  # Shape: (num_samples, 16)
# y = [...]  # Shape: (num_samples,) with values from 0 to 11

# # Convert to torch tensors
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.long)

# # Split the data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

# # Define the Logistic Regression Model
# class LogisticRegressionModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LogisticRegressionModel, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

#     def forward(self, x):
#         return self.fc(x)  # No activation, as we will use CrossEntropyLoss which applies softmax internally

# # Initialize the model
# input_dim = 16  # Number of features per input vector
# output_dim = 12  # Number of classes
# model = LogisticRegressionModel(input_dim, output_dim)

# # Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()  # This loss function applies softmax internally
# optimizer = optim.SGD(model.parameters(), lr=0.01)  # Use SGD optimizer

# # Training the model
# num_epochs = 100  # Number of epochs to train
# batch_size = 32  # Batch size for training

# for epoch in range(num_epochs):
#     model.train()  # Set the model to training mode

#     # Shuffle and iterate over the dataset in batches
#     permutation = torch.randperm(X_train.size(0))
#     for i in range(0, X_train.size(0), batch_size):
#         indices = permutation[i:i+batch_size]
#         batch_X, batch_y = X_train[indices], y_train[indices]

#         # Forward pass
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)

#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     # Print loss every 10 epochs
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# # Evaluate the model
# model.eval()  # Set the model to evaluation mode
# with torch.no_grad():
#     outputs = model(X_test)
#     _, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test).sum().item() / y_test.size(0)
#     print(f"Test Accuracy: {accuracy*100:.2f}%")

