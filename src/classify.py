import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

    def forward(self, x):
        return self.fc(x)  # No activation, as we will use CrossEntropyLoss which applies softmax internally

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        # self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Second hidden layer
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(hidden_dim, output_dim)  # Output layer

        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Second hidden layer


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        return x
    


def evaluate_model(model, X, y, model_name):
    """Evaluates the model on data and prints correct predictions."""

    checkpoint = torch.load(f'{PATH_PREFIX}/classify/{model_name}.pt', weights_only=False)
    model.load_state_dict(checkpoint["model"])

    model.eval()  # Set the model to evaluation mode
    correct_predictions = []
    mod_20_counts = {i: 0 for i in range(20)}

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        
        for idx, (pred, true) in enumerate(zip(predicted, y)):
            if pred == true:
                correct_predictions.append((idx, pred.item()))
                mod_20_value = idx % 20
                mod_20_counts[mod_20_value] += 1

        accuracy = len(correct_predictions) / y.size(0)
    
    # print("Correctly predicted values and their indices:")
    # for idx, value in correct_predictions:
    #     print(f"Index: {idx}, mod {idx % 20}, Predicted Value: {value}")
    
    print("mod 20 counts: ", mod_20_counts)
    print(f"\nTotal correct predictions: {len(correct_predictions)}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

def prepare_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """Splits the data into train, validation and test sets."""
    # First split into train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Then split train+val into train and val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, random_state=random_state)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_model(model, train_data, val_data, criterion, optimizer, num_epochs=100, batch_size=32, patience=10, model_name=None):
    """Trains the model using validation accuracy for early stopping."""
    X_train, y_train = train_data
    X_val, y_val = val_data

    wandb.init(
        project="classify-card",
        config={
            "epochs": num_epochs,
            "batch_size": batch_size,
            "patience": patience,
        },
        name=model_name
    )

    counter = 0
    best_val_loss = 1e10
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        permutation = torch.randperm(X_train.size(0))
        train_loss = 0
        
        for i in range(0, X_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_X, batch_y = X_train[indices], y_train[indices]
    
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / (X_train.size(0) // batch_size)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        })
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            counter = 0
            best_val_loss = val_loss

            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch_num": epoch,
                "best_val_loss": best_val_loss,
            }
            torch.save(checkpoint, f'{PATH_PREFIX}/classify/{model_name}.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

def run_classify(X, y, model_name, input_dim=16, output_dim=12, num_epochs=100, batch_size=32, lr=0.001, model_type="linear"):
    """Main function to run the model training and evaluation."""
    # Prepare data with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    if model_type == "linear":
        model = LinearModel(input_dim, output_dim).to(device)
    elif model_type == "mlp":
        model = MLPModel(input_dim=input_dim, hidden_dim=32, output_dim=output_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model with validation data
    train_model(
        model, 
        (X_train, y_train),
        (X_val, y_val),
        criterion, 
        optimizer, 
        num_epochs, 
        batch_size, 
        model_name=f"{model_name}_{model_type}"
    )

    # Evaluate the model
    train_accuracy = evaluate_model(model, X_train, y_train, model_name=f"{model_name}_{model_type}")
    val_accuracy = evaluate_model(model, X_val, y_val, model_name=f"{model_name}_{model_type}")
    test_accuracy = evaluate_model(model, X_test, y_test, model_name=f"{model_name}_{model_type}")

    wandb.log({
        "final_train_accuracy": train_accuracy,
        "final_val_accuracy": val_accuracy,
        "final_test_accuracy": test_accuracy
    })
    wandb.finish()

    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


def prepare_paired_data(card_embeddings, attr_embeddings, labels):
    """Prepare paired embeddings data for binary classification."""
    # Convert to torch tensors if they aren't already
    if not isinstance(card_embeddings, torch.Tensor):
        card_embeddings = torch.tensor(card_embeddings, dtype=torch.float32)
    if not isinstance(attr_embeddings, torch.Tensor):
        attr_embeddings = torch.tensor(attr_embeddings, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    # Split into train and test sets
    indices = torch.randperm(len(card_embeddings))
    split = int(0.8 * len(card_embeddings))
    train_indices = indices[:split]
    test_indices = indices[split:]

    X_train_cards = card_embeddings[train_indices]
    X_train_attrs = attr_embeddings[train_indices]
    y_train = labels[train_indices]

    X_test_cards = card_embeddings[test_indices]
    X_test_attrs = attr_embeddings[test_indices]
    y_test = labels[test_indices]

    return (X_train_cards, X_train_attrs, y_train), (X_test_cards, X_test_attrs, y_test)

class BinaryPairClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(BinaryPairClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim * 2, 1)  # Concatenate two embeddings
        self.sigmoid = nn.Sigmoid()

    def forward(self, card_embedding, attr_embedding):
        # Concatenate the two embeddings
        combined = torch.cat((card_embedding, attr_embedding), dim=1)
        output = self.fc(combined)
        return self.sigmoid(output)

def train_binary_model(model, train_data, criterion, optimizer, num_epochs, batch_size, model_name):
    """Train the binary classifier."""
    X_train_cards, X_train_attrs, y_train = train_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    n_samples = len(X_train_cards)
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Process in batches
        for i in range(0, n_samples, batch_size):
            batch_cards = X_train_cards[i:i+batch_size].to(device)
            batch_attrs = X_train_attrs[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_cards, batch_attrs).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (n_samples // batch_size)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        wandb.log({"train_loss": avg_loss})
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'{PATH_PREFIX}/classify/{model_name}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break

def evaluate_binary_model(model, test_data, model_name):
    """Evaluate the binary classifier."""
    X_test_cards, X_test_attrs, y_test = test_data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_test_cards = X_test_cards.to(device)
        X_test_attrs = X_test_attrs.to(device)
        y_test = y_test.to(device)
        
        outputs = model(X_test_cards, X_test_attrs).squeeze()
        predicted = (outputs >= 0.5).float()
        
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        
        print(f'Test Accuracy: {accuracy:.4f}')
        return accuracy

def run_binary_classify(card_embeddings, attr_embeddings, labels, model_name, 
                       embedding_dim=64, num_epochs=100, batch_size=32, lr=0.001):
    """Main function to run binary classification training and evaluation."""
    # Prepare data
    train_data, test_data = prepare_paired_data(card_embeddings, attr_embeddings, labels)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, loss function, and optimizer
    model = BinaryPairClassifier(embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train the model
    train_binary_model(model, train_data, criterion, optimizer, num_epochs, batch_size, model_name)
    
    # Evaluate the model
    train_accuracy = evaluate_binary_model(model, train_data, model_name)
    test_accuracy = evaluate_binary_model(model, test_data, model_name)
    
    wandb.log({"train_accuracy": train_accuracy, "test_accuracy": test_accuracy})
    wandb.finish()
    
    return train_accuracy, test_accuracy


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
# class LinearModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(LinearModel, self).__init__()
#         self.fc = nn.Linear(input_dim, output_dim)  # Linear layer (16 -> 12)

#     def forward(self, x):
#         return self.fc(x)  # No activation, as we will use CrossEntropyLoss which applies softmax internally

# # Initialize the model
# input_dim = 16  # Number of features per input vector
# output_dim = 12  # Number of classes
# model = LinearModel(input_dim, output_dim)

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

