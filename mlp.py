import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd


DATA_DIR = '/Users/aplle/Code/MathModeling/SC/dataset1'
MODEL_DIR = '/Users/aplle/Code/MathModeling/SC/models'

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
eval_df = pd.read_csv(f"{DATA_DIR}/eval.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

POWER_DIVISOR = 1000.0
YEAR_MIN = 2016
YEAR_MAX = 2018

def is_leap(yr:int):
    if yr % 400 == 0:
        return True
    elif yr % 100 == 0:
        return False
    elif yr % 4 == 0:
        return True
    return False

def get_ndays(yr:int):
    return 366 if is_leap(yr) else 365

def preprocess_df(df):
    df.dropna(subset=['Power (kW)'], inplace=True)
    df['Power (kW)'] = df['Power (kW)'] / POWER_DIVISOR
    df.rename(columns={'Power (kW)': 'Power'}, inplace=True)

    df['Days_from_NYD'] = df['Days_from_NYD'] / df['Year'].apply(get_ndays)

    df['Year'] = df['Year'] - YEAR_MIN
    df['Year'] = df['Year'] / (YEAR_MAX - YEAR_MIN)

    # BLIND THE YEAR INFORMATION
    # df['Year'] = 0.0

    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Time'] = df['Time'].dt.hour * 60 + df['Time'].dt.minute
    df['Time'] = df['Time'] / (24 * 60)

    df.drop(['Day', 'Span'], axis=1, inplace=True)

    df['Month'] = df['Month'] / 12.0

    return pd.get_dummies(df, columns=['Weekday', 'Region'], drop_first=True)


train_df = preprocess_df(train_df)
# keep canonical columns from train
_train_columns = train_df.columns.copy()

eval_df = preprocess_df(eval_df)
# reindex eval to have same columns as train (fill missing dummies with 0)
eval_df = eval_df.reindex(columns=_train_columns, fill_value=0)

test_df = preprocess_df(test_df)
# reindex test to have same columns as train
test_df = test_df.reindex(columns=_train_columns, fill_value=0)



feature_cols = train_df.drop(columns=['Power']).columns
input_dim = len(feature_cols)

X_train = torch.tensor(train_df[feature_cols].to_numpy().astype(np.float64), dtype=torch.float32)
y_train = torch.tensor(train_df['Power'].to_numpy().astype(np.float64).reshape(-1, 1), dtype=torch.float32)
train_dataset = data.TensorDataset(X_train, y_train)

X_eval = torch.tensor(eval_df[feature_cols].to_numpy().astype(np.float64), dtype=torch.float32)
y_eval = torch.tensor(eval_df['Power'].to_numpy().astype(np.float64).reshape(-1, 1), dtype=torch.float32)
eval_dataset = data.TensorDataset(X_eval, y_eval)

X_test = torch.tensor(test_df[feature_cols].to_numpy().astype(np.float64), dtype=torch.float32)
y_test = torch.tensor(test_df['Power'].to_numpy().astype(np.float64).reshape(-1, 1), dtype=torch.float32)
test_dataset = data.TensorDataset(X_test, y_test)



mlp = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    # nn.Dropout(0.1),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(32, 8),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(8, 1)
)



def train_iter(model, dataset, opt, loss, batch_size=32):
    model.train()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_loss = 0.0
    for X_batch, y_batch in loader:
        opt.zero_grad()
        y_pred = model(X_batch)
        l = loss(y_pred, y_batch)
        l.backward()
        opt.step()
        total_loss += l.item() * X_batch.size(0)
    return total_loss / len(dataset)

def evaluate(model, dataset, loss, batch_size=32):
    model.eval()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            l = loss(y_pred, y_batch)
            total_loss += l.item() * X_batch.size(0)
    return total_loss / len(dataset)

def train(model, train_dataset, eval_dataset, epochs=100, batch_size=32, lr=1e-3, weight_decay=0.0, patience=10, checkpoint_path=None):
    """Train with simple early stopping and optional checkpointing.

    Args:
        model: torch.nn.Module
        train_dataset, eval_dataset: Dataset
        epochs, batch_size, lr, weight_decay: optimizer params
        patience: stop if eval loss doesn't improve for this many epochs
        checkpoint_path: if provided, save best model to this path

    Returns:
        best_eval_loss
    """
    import copy
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_eval = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        train_loss = train_iter(model, train_dataset, opt, loss_fn, batch_size)
        eval_loss = evaluate(model, eval_dataset, loss_fn, batch_size)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Eval Loss: {eval_loss:.6f}")

        if eval_loss < best_eval - 1e-9:
            best_eval = eval_loss
            epochs_no_improve = 0
            # store best weights in memory
            best_state = copy.deepcopy(model.state_dict())
            if checkpoint_path:
                try:
                    torch.save(best_state, checkpoint_path)
                except Exception:
                    pass
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f"Early stopping after {epoch+1} epochs (no improvement for {patience} epochs). Best eval: {best_eval:.6f}")
            break

    # restore best weights if we have them
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass

    return best_eval




train(mlp, train_dataset, eval_dataset, epochs=50, batch_size=32, lr=1e-4, weight_decay=1e-6, patience=5, checkpoint_path=os.path.join(MODEL_DIR,'mlp_checkpoint.pth'))



def unembedding(model, X):
    model.eval()
    with torch.no_grad():
        return model(X) * POWER_DIVISOR


def test(model, test_dataset=test_dataset, max_count=None):
    """Return a DataFrame with true and predicted power (kW) for up to max_count examples.

    This function uses the tensors already created earlier (X_test/y_test stored in the notebook) via the provided dataset.
    """
    # Build a loader so we can slice the first `max_count` examples safely.
    loader = data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

    X_parts = []
    y_parts = []
    seen = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            if max_count is None:
                X_parts.append(X_batch)
                y_parts.append(y_batch)
            else:
                need = max_count - seen
                if need <= 0:
                    break
                if X_batch.size(0) > need:
                    X_parts.append(X_batch[:need])
                    y_parts.append(y_batch[:need])
                    seen += need
                    break
                else:
                    X_parts.append(X_batch)
                    y_parts.append(y_batch)
                    seen += X_batch.size(0)

    if len(X_parts) == 0:
        return pd.DataFrame(columns=['Power (kW)', 'Predicted Power (kW)'])

    X_all = torch.cat(X_parts, dim=0)
    y_all = torch.cat(y_parts, dim=0)

    y_pred = unembedding(model, X_all).squeeze().cpu().numpy()
    y_true = (y_all.squeeze().cpu().numpy()) * POWER_DIVISOR

    ret = pd.DataFrame({'Power (kW)': y_true, 'Predicted Power (kW)': y_pred})
    return ret



test(mlp, eval_dataset, max_count=50)