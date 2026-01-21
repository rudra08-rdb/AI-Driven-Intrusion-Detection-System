import torch
import numpy as np

from model_definitions import IDSModelHybrid

# ------------------ CONFIG ------------------
MODEL_PATH = "trained_ids_model.pth"

NUM_FEATURES = 78
NUM_CLASSES = 15
SEQ_LEN = 1


# ------------------ LOAD MODEL ------------------
def load_model():
    """
    Loads the trained IDS model safely using state_dict.
    Compatible with Streamlit Cloud.
    """
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


# ------------------ PREDICTION ------------------
@torch.no_grad()
def predict(model, feature_vector):
    """
    Predicts attack class and confidence for a single flow.

    Args:
        model: Loaded IDSModelHybrid
        feature_vector: numpy array of shape (78,)

    Returns:
        dict with prediction index and confidence
    """

    # Convert input to tensor
    x = torch.tensor(
        feature_vector,
        dtype=torch.float32
    ).view(1, SEQ_LEN, NUM_FEATURES)

    # Forward pass
    logits, _, _ = model(x)

    probs = torch.softmax(logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    return {
        "prediction": int(pred_class.item()),
        "confidence": float(confidence.item())
    }



