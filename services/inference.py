import torch
import os
from model_definitions import IDSModelHybrid

# ------------------ CONFIG ------------------
MODEL_PATH = "trained_ids_model.pth"
NUM_FEATURES = 78
NUM_CLASSES = 15
SEQ_LEN = 1

# ------------------ LOAD MODEL ------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}"
        )

    # Rebuild model architecture
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    # Load FULL model safely
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    # If full model was saved
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


# ------------------ PREDICT ------------------
def predict(model, features):
    x = torch.tensor(features, dtype=torch.float32)
    x = x.view(1, 1, -1)  # (batch, seq, features)

    with torch.no_grad():
        logits, _, _ = model(x)
        probs = torch.softmax(logits, dim=1)

    pred = int(torch.argmax(probs, dim=1).item())
    confidence = float(torch.max(probs).item())

    return {
        "prediction": pred,
        "confidence": confidence
    }


