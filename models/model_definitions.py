import torch
import numpy as np
from model_definitions import IDSModelHybrid

# ---------------- CONFIG ----------------
SEQ_LEN = 1               # single flow treated as 1 timestep
FEAT_DIM = 78             # CICIDS numeric features
NUM_CLASSES = 15
MODEL_PATH = "trained_ids_model.pth"

# ---------------- LOAD MODEL ----------------
def load_model():
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=FEAT_DIM,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ---------------- PREDICT ----------------
def predict(model, feature_vector):
    """
    feature_vector: numpy array of shape (78,)
    """

    x = np.array(feature_vector, dtype=np.float32)
    x = x.reshape(1, SEQ_LEN, FEAT_DIM)  # (batch, seq, feat)

    x_tensor = torch.tensor(x)

    with torch.no_grad():
        logits, _, attn = model(x_tensor)
        probs = torch.softmax(logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "attention": attn
    }
