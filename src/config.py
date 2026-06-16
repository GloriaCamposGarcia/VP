import os
from pathlib import Path
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Definir la raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cargar variables de entorno
if HAS_DOTENV:
    load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

# Rutas de datos
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")

# Asegurar que las carpetas existan
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Directorios de entrenamiento, uso y compartidos (MLOps / Compliance)
SHARED_DIR = DATA_PROCESSED_DIR / "shared"
TRAIN_RUNS_DIR = DATA_PROCESSED_DIR / "train" / "runs"
USE_RUNS_DIR = DATA_PROCESSED_DIR / "use" / "runs"

SHARED_DIR.mkdir(parents=True, exist_ok=True)
(SHARED_DIR / "models").mkdir(parents=True, exist_ok=True)
TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
USE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Modo de ejecución por defecto: 'use' o 'train'
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "use").strip().lower()

RUN_DATE = datetime.now().strftime('%Y-%m-%d')

if PIPELINE_MODE == "train":
    RUN_DIR = TRAIN_RUNS_DIR / f"run_{RUN_DATE}"
else:
    RUN_DIR = USE_RUNS_DIR / f"run_{RUN_DATE}"

RUN_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros de embeddings
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'pld_aml_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("AML_System")

logger.info(f"Configuración cargada. Backend de embeddings: {EMBEDDING_BACKEND}")
