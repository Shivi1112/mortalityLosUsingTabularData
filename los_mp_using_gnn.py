import numpy as np
import pandas as pd
import sklearn as sk
import os
import numpy as np
import pandas as pd
import sklearn as sk
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    classification_report, 
    mean_squared_error, 
    r2_score,
    confusion_matrix
)
import json
import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import pickle



folder = "mimic"
files = os.listdir(folder)
csv_files = [f for f in files if f.endswith(".csv.gz")]


# In[22]:


for files in csv_files:
  print (files)



merged_data = pd.read_csv(os.path.join(folder, "preprocess.csv" ))


diagnoses_icd = pd.read_csv(os.path.join(folder, "DIAGNOSES_ICD.csv.gz" ), compression="gzip")
diagnoses_icd.columns = diagnoses_icd.columns.str.lower()
diagnoses_icd.head(5)


merged_data = pd.merge(merged_data , diagnoses_icd , on=["subject_id", "hadm_id"])
merged_data


# In[32]:



def create_graph_data_with_masks(df, test_size=0.2, random_state=42):
    """
    Create graph data with train/test masks
    """
    # Prepare features and targets
    X_numeric = df.drop(['hospital_expire_flag', 'los'], axis=1).select_dtypes(include=[np.number])
    y_mortality = df['hospital_expire_flag']
    y_los = df['los']

    # Scale features
    scaler = StandardScaler()
    node_features = scaler.fit_transform(X_numeric)

    # Create edge index based on k-nearest neighbors
    similarity_matrix = cosine_similarity(node_features)
    edge_index = []
    k = 5  # Number of nearest neighbors
    for i in range(similarity_matrix.shape[0]):
        neighbors = np.argsort(similarity_matrix[i])[-k-1:-1]
        for j in neighbors:
            edge_index.append([i, j])
            edge_index.append([j, i])  # Add reverse edge for undirected graph

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Create train/test masks
    num_nodes = len(df)
    indices = np.arange(num_nodes)
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_state)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    y_mortality = torch.tensor(y_mortality.values, dtype=torch.long)
    y_los = torch.tensor(y_los.values, dtype=torch.float32)

    # Verify data
    assert not torch.isnan(x).any(), "Features contain NaN values"
    assert not torch.isnan(edge_index).any(), "Edge indices contain NaN values"
    assert edge_index.max() < num_nodes, "Edge index contains invalid node references"

    # Create Data object with masks
    data = Data(
        x=x,
        edge_index=edge_index,
        y_mortality=y_mortality,
        y_los=y_los,
        train_mask=train_mask,
        test_mask=test_mask,
        num_nodes=num_nodes
    )

    return data, scaler



def preprocess_healthcare_data(df):
    """
    Comprehensive preprocessing of healthcare data with ICD code aggregation.
    """
    df = df.copy()

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()
  
    df=df.drop(['admittime','dischtime','discharge_location','diagnosis','religion','ethnicity','insurance'],axis=1)
    # Fill categorical columns with most frequent values
    categorical_cols = ['admission_type', 'admission_location', 'language', 'marital_status', 'gender']

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Create dummy variables for categorical columns
    dummies_icd9 = pd.get_dummies(df['icd9_code'], prefix='icd9')
    dummies_categorical = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)

    # Step 2: Concatenate dummies with subject_id, hadm_id, los, and hospital_expire_flag
    columns_to_keep = ['subject_id', 'hadm_id', 'los']
    if 'hospital_expire_flag' in df.columns:
        columns_to_keep.append('hospital_expire_flag')

    df = pd.concat([df[columns_to_keep], dummies_icd9, dummies_categorical], axis=1)

    # Step 3: Group by subject_id and hadm_id, keeping first value for non-dummy variables
    agg_dict = {
        'los': 'first',
        'hospital_expire_flag': 'first' if 'hospital_expire_flag' in df.columns else None
    }
    # Add sum aggregation for all dummy columns
    agg_dict.update({col: 'sum' for col in df.columns
                    if col not in ['subject_id', 'hadm_id', 'los', 'hospital_expire_flag']})

    # Remove None values from agg_dict
    agg_dict = {k: v for k, v in agg_dict.items() if v is not None}

    df_grouped = df.groupby(['subject_id', 'hadm_id']).agg(agg_dict).reset_index()

    # Summary statistics for LOS and mortality
    print("\nLength of Stay (LOS) Statistics:")
    print(df_grouped['los'].describe())

    if 'hospital_expire_flag' in df_grouped.columns:
        mortality_rate = df_grouped['hospital_expire_flag'].mean() * 100
        print(f"\nMortality Rate: {mortality_rate:.2f}%")
        print(f"Total Deaths: {df_grouped['hospital_expire_flag'].sum()}")
        print(f"Total Admissions: {len(df_grouped)}")
    print(df_grouped.shape,df_grouped.columns)
    return df_grouped

class GATModel(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_heads=2):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=num_heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        self.mortality_out = nn.Linear(hidden_dim, 1)  # Binary classification
        self.los_out = nn.Linear(hidden_dim, 4)  # 4-class classification
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        assert not torch.isnan(x).any(), "Input features contain NaN values"
        assert not torch.isnan(edge_index).any(), "Edge index contains NaN values"

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        mortality_pred = self.mortality_out(x)
        los_pred = self.los_out(x)
        return mortality_pred, los_pred

def train_and_evaluate(model, data, device, epochs=500, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    # Calculate class weights for mortality
    train_labels = data.y_mortality[data.train_mask].cpu().numpy()
    class_counts = np.bincount(train_labels)
    pos_weight = torch.tensor([class_counts[0]/class_counts[1]]).to(device)

    # Loss functions
    mortality_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    los_criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss for multi-class

    best_val_loss = float('inf')
    counter = 0
    best_model_state = None

    try:
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()

            mortality_logits, los_logits = model(data.x, data.edge_index)

            # Calculate losses
            loss_mortality = mortality_criterion(
                mortality_logits[data.train_mask].squeeze(),
                data.y_mortality[data.train_mask].float()
            )
            loss_los = los_criterion(
                los_logits[data.train_mask],
                data.y_los[data.train_mask].long()  # Convert to long for CrossEntropyLoss
            )

            loss = loss_mortality + loss_los
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_mortality_logits, val_los_logits = model(data.x, data.edge_index)

                    val_loss_mortality = mortality_criterion(
                        val_mortality_logits[data.test_mask].squeeze(),
                        data.y_mortality[data.test_mask].float()
                    )
                    val_loss_los = los_criterion(
                        val_los_logits[data.test_mask],
                        data.y_los[data.test_mask].long()
                    )
                    val_loss = val_loss_mortality + val_loss_los

                    # Calculate validation metrics
                    val_mortality_pred = (torch.sigmoid(val_mortality_logits[data.test_mask]) > 0.5).float()
                    val_los_pred = torch.argmax(val_los_logits[data.test_mask], dim=1)

                    mortality_accuracy = (val_mortality_pred.squeeze() == data.y_mortality[data.test_mask]).float().mean()
                    los_accuracy = (val_los_pred == data.y_los[data.test_mask].long()).float().mean()

                    print(f'Epoch {epoch+1:03d}, '
                          f'Train Loss: {loss.item():.4f}, '
                          f'Val Loss: {val_loss.item():.4f}, '
                          f'Val Mortality Acc: {mortality_accuracy:.4f}, '
                          f'Val LOS Acc: {los_accuracy:.4f}')

                    # Early stopping logic
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        counter = 0
                    else:
                        counter += 1
                        if counter >= patience:
                            print("Early stopping triggered")
                            break
                model.train()

    except RuntimeError as e:
        print(f"Error during training: {e}")
        return

    # Final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        mortality_logits, los_logits = model(data.x, data.edge_index)

        # Mortality evaluation
        mortality_probs = torch.sigmoid(mortality_logits)
        test_mortality_pred = (mortality_probs[data.test_mask] > 0.5).float()
        test_mortality_true = data.y_mortality[data.test_mask]

        # LOS evaluation
        test_los_pred = torch.argmax(los_logits[data.test_mask], dim=1)
        test_los_true = data.y_los[data.test_mask].long()

        # Convert to CPU for metrics calculation
        test_mortality_pred_cpu = test_mortality_pred.cpu().numpy()
        test_mortality_true_cpu = test_mortality_true.cpu().numpy()
        test_los_pred_cpu = test_los_pred.cpu().numpy()
        test_los_true_cpu = test_los_true.cpu().numpy()

        # Calculate metrics
        mortality_metrics = {
            'accuracy': accuracy_score(test_mortality_true_cpu, test_mortality_pred_cpu),
            'precision': precision_score(test_mortality_true_cpu, test_mortality_pred_cpu),
            'recall': recall_score(test_mortality_true_cpu, test_mortality_pred_cpu),
            'f1': f1_score(test_mortality_true_cpu, test_mortality_pred_cpu)
        }

        los_metrics = {
            'accuracy': accuracy_score(test_los_true_cpu, test_los_pred_cpu),
            'macro_f1': f1_score(test_los_true_cpu, test_los_pred_cpu, average='macro'),
            'weighted_f1': f1_score(test_los_true_cpu, test_los_pred_cpu, average='weighted')
        }

        print("\n=== Mortality Prediction Results ===")
        for metric, value in mortality_metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}')

        print("\n=== Length of Stay Prediction Results ===")
        for metric, value in los_metrics.items():
            print(f'{metric.capitalize()}: {value:.4f}')
        print("\nLOS Classification Report:")
        print(classification_report(test_los_true_cpu, test_los_pred_cpu))

        return {**mortality_metrics, **los_metrics}
    
    
def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading data...")
    df = merged_data
    print(df.shape)

    print("\nPreprocessing data...")
    df_processed = preprocess_healthcare_data(df)  # Make sure this returns both df and scaler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create graph data with train/test masks
    data,scaler = create_graph_data_with_masks(df_processed)
    print("Graph data:", data)

    # Initialize model
    input_dim = data.x.shape[1]
    model = GATModel(input_dim).to(device)
    data = data.to(device)

    # Train and evaluate
    results = train_and_evaluate(model, data, device)

    if results:
        # Create directory for saving models and results
        save_dir = 'gat_models'
        os.makedirs(save_dir, exist_ok=True)

        # Save model and scaler
        save_model_and_scaler(model, scaler, save_dir)

        # Save results
        results_path = os.path.join(save_dir, 'gnn_results.json')
        metrics = {
            "mortality_accuracy": float(results['accuracy']),  # Updated to match new metrics
            "mortality_f1": float(results['f1']),
            "los_accuracy": float(results['accuracy']),
            "los_macro_f1": float(results['macro_f1'])
        }

        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Results saved to {results_path}")

        # Save model configuration
        config_path = os.path.join(save_dir, 'model_config.json')
        config = {
            "model_type": "GATModel",
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_heads": 2,
            "training_params": {
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "weight_decay": 5e-4,
                "epochs": 500,
                "early_stopping_patience": 20
            },
            "device": str(device),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        }

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Model configuration saved to {config_path}")

def save_model_and_scaler(model, scaler, save_dir='gat_models'):
    """
    Save both the model and scaler to the specified directory

    Args:
        model: The trained GAT model
        scaler: The fitted StandardScaler
        save_dir: Directory to save the files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(save_dir, 'gat_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': model.conv1.in_channels,
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }, model_path)

    # Save scaler
    scaler_path = os.path.join(save_dir, 'standard_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

# Function to load the saved model and scaler
def load_model_and_scaler(save_dir='gat_models', device='cuda'):
    """
    Load the saved model and scaler

    Args:
        save_dir: Directory containing the saved files
        device: Device to load the model to

    Returns:
        model: The loaded model
        scaler: The loaded scaler
    """
    model_path = os.path.join(save_dir, 'gat_model.pt')
    scaler_path = os.path.join(save_dir, 'standard_scaler.pkl')

    try:
        # Load model
        checkpoint = torch.load(model_path)
        model = GATModel(checkpoint['input_dim']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        print(f"Model and scaler loaded successfully")
        return model, scaler

    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None
    
if __name__ == "__main__":
    main()

