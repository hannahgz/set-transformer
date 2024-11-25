import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_utils import initialize_loaders
from model import GPT
import os
from itertools import permutations
from data_utils import split_data

PATH_PREFIX = '/n/holylabs/LABS/wattenberg_lab/Lab/hannahgz_tmp'



def construct_binding_id_dataset(config, dataset_name, capture_layer):

    perms = list(permutations(range(20), 2))

    dataset_path = f"{PATH_PREFIX}/{dataset_name}.pth"
    dataset = torch.load(dataset_path)
    train_loader, val_loader = initialize_loaders(config, dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(config).to(device)

    X = []
    y = []

    for index, batch in enumerate(val_loader):
        print(f"Batch {index}/{len(val_loader)}")
        print("X len: ", len(X))
        print("y len: ", len(y))
        batch = batch.to(device)
        _, _, _, captured_embedding = model(batch, True, capture_layer)
        # torch.Size([64, 49, 64])
        # [batch_size, seq_len, embedding_dim]

        for index, indiv_embedding in enumerate(captured_embedding):
            curr_embedding = indiv_embedding[:40]
            curr_tokens = batch[index][:40]
            for (element1_index, element2_index) in perms:            
                element1 = curr_embedding[element1_index * 2 + 1]
                element2 = curr_embedding[element2_index * 2 + 1]

                token1 = curr_tokens[element1_index * 2]
                token2 = curr_tokens[element2_index * 2]

                X.append(torch.cat((element1, element2)))
                if (token1 == token2):
                    y.append(1)
                else:
                    y.append(0)


    base_dir = f"{PATH_PREFIX}/binding_id/{dataset_name}/layer{capture_layer}"
    os.makedirs(base_dir, exist_ok=True)

    X_path = os.path.join(base_dir, "X.pt")
    y_path = os.path.join(base_dir, "y.pt")

    X_tensor = torch.stack(X)
    y_tensor = torch.tensor(y)

    breakpoint()
    torch.save(X_tensor, X_path)
    torch.save(y_tensor, y_path)

    return X_tensor, y_tensor


def train_binding_classifier(X, y, model_name, input_dim=128, num_epochs=100, batch_size=32, lr=0.001, patience=10):
    """Train a binary classifier for binding identification."""
    # Initialize wandb
    wandb.init(project="binding-id-classifier", name=model_name)

    # Prepare data with validation split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, loss function, and optimizer
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # Training loop
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size].to(device)
            batch_y = y_train[i:i+batch_size].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / (len(X_train) / batch_size)
        
        # Validation loop
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device)).squeeze()
            val_loss = criterion(val_outputs, y_val.float().to(device))
            
            # Calculate accuracies
            train_preds = (model(X_train.to(device)).squeeze() > 0.5).float()
            val_preds = (val_outputs > 0.5).float()
            
            train_acc = (train_preds == y_train.float().to(device)).float().mean()
            val_acc = (val_preds == y_val.float().to(device)).float().mean()
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_loss.item(),
            'train_acc': train_acc.item(),
            'val_acc': val_acc.item()
        })
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save best model
            torch.save(model.state_dict(), f'{PATH_PREFIX}/binding_id/{model_name}_best.pt')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device)).squeeze()
        test_preds = (test_outputs > 0.5).float()
        test_acc = (test_preds == y_test.float().to(device)).float().mean()
        
    print(f"Final Test Accuracy: {test_acc.item()*100:.2f}%")
    wandb.log({'test_accuracy': test_acc.item()})
    wandb.finish()
    
    return model
