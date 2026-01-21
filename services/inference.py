import torch
from model_definitions import IDSModelHybrid

MODEL_PATH = "trained_ids_model.pth"

NUM_FEATURES = 78
NUM_CLASSES = 15
SEQ_LEN = 1

def load_model():
    model = IDSModelHybrid(
        seq_len=SEQ_LEN,
        feat_dim=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        use_attention=True
    )

    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


