import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = Path("thyroid_processed_data_cleaned.csv")
K = 50  # top anomalies to highlight
SEED = 42
EPOCHS = 50
BATCH_SIZE = 256
LR = 1e-3


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_data(path: Path):
    df = pd.read_csv(path)
    # Map labels to 0/1 if strings are used
    if df["Outlier_label"].dtype == object:
        df["Outlier_label"] = df["Outlier_label"].map({"n": 0, "o": 1})
    y = df["Outlier_label"].to_numpy()
    X = df.drop(columns=["Outlier_label"])
    return X, y


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out


def evaluate(y_true, scores, k=50):
    roc = roc_auc_score(y_true, scores)
    pr = average_precision_score(y_true, scores)
    topk = np.mean(y_true[np.argsort(-scores)[:k]])
    return {"ROC-AUC": roc, "PR-AUC": pr, f"Precision@{k}": topk}


def main():
    set_seed(SEED)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    X_df, y = load_data(DATA_PATH)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = AutoEncoder(input_dim=X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss()

    # Simple mini-batch training
    model.train()
    n = X_tensor.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            batch = X_tensor[idx]
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= n
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - loss: {epoch_loss:.6f}")

    # Inference
    model.eval()
    with torch.no_grad():
        recon_all = model(X_tensor)
        recon_err = torch.mean((recon_all - X_tensor) ** 2, dim=1).cpu().numpy()

    metrics = evaluate(y, recon_err, k=K)
    print("Autoencoder reconstruction performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Threshold using 97th percentile (aligning with ~3% contamination)
    threshold = float(np.quantile(recon_err, 0.97))
    y_pred = (recon_err >= threshold).astype(int)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["True_Normal", "True_Anomaly"], columns=["Pred_Normal", "Pred_Anomaly"])
    print(f"Threshold (97th percentile): {threshold:.6f}")
    print(cm_df)

    # Save scores
    df_scores = pd.DataFrame({"anomaly_score_ae": recon_err, "Outlier_label": y})
    out_csv = Path("ae_reconstruction_scores.csv")
    df_scores.to_csv(out_csv, index=False)
    print(f"Saved AE scores to {out_csv}")

    # Visualization: PCA colored by recon error
    pca = PCA(n_components=2, random_state=SEED)
    X_pca = pca.fit_transform(X_scaled)
    top_idx = np.argsort(-recon_err)[:K]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=recon_err, s=10, alpha=0.7, cmap="viridis")
    plt.scatter(
        X_pca[top_idx, 0],
        X_pca[top_idx, 1],
        facecolors="none",
        edgecolors="red",
        s=60,
        linewidths=1.5,
        label=f"Top {K} anomalies",
    )
    plt.colorbar(sc, label="Reconstruction error (MSE)")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title("PCA colored by AE reconstruction error")
    plt.legend()
    plt.tight_layout()
    fig_path = Path("ae_pca_topK.png")
    plt.savefig(fig_path, dpi=300)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
