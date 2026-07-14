import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
KNOWLEDGE_DIR   = os.path.join(BASE_DIR, "knowledge_docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

HAM_IMAGES_DIR  = os.path.join(RAW_DIR, "ham10000_images")
HAM_CSV_PATH    = os.path.join(RAW_DIR, "HAM10000_metadata.csv")

MODEL_PATH      = os.path.join(MODEL_DIR, "efficientnet_skin.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "efficientnet_skin_best.pth")

# ── Dataset ────────────────────────────────────────────────────────────────────
RANDOM_SEED       = 42
TEST_SIZE         = 0.15
VAL_SIZE          = 0.15
NUM_WORKERS       = 0        # Windows safe
USE_CLASS_WEIGHTS = True

CLASS_NAMES = [
    "Actinic Keratoses (akiec)",
    "Basal Cell Carcinoma (bcc)",
    "Benign Keratosis-like Lesions (bkl)",
    "Dermatofibroma (df)",
    "Melanoma (mel)",
    "Melanocytic Nevi (nv)",
    "Vascular Lesions (vasc)",
]

CLASS_LABELS = {
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
}

IDX_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}
NUM_CLASSES  = 7

# ── Image ──────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
MEAN       = [0.7630, 0.5456, 0.5700]   # HAM10000 channel means
STD        = [0.1409, 0.1521, 0.1697]   # HAM10000 channel stds

# ── Training ───────────────────────────────────────────────────────────────────
MODEL_ARCH      = "efficientnet_b3"
EPOCHS          = 30
BATCH_SIZE      = 8        # safe for RTX 2050 4 GB
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-5
SCHEDULER       = "cosine"
DROPOUT         = 0.3
FREEZE_BACKBONE = False    # fine-tune all layers

# ── Grad-CAM ───────────────────────────────────────────────────────────────────
GRADCAM_ALPHA = 0.4

# ── RAG ────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 64
TOP_K_RETRIEVAL = 5

# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL",    "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MAX_TOKENS   = 1024
TEMPERATURE  = 0.3

# ── API ────────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
