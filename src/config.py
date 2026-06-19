import os
from pathlib import Path
import logging
from datetime import datetime

try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

# Se determina el directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Se cargan las variables de entorno si la biblioteca dotenv está disponible
if HAS_DOTENV:
    load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

# Se definen las rutas base para los conjuntos de datos
DATA_RAW_DIR = PROJECT_ROOT / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = PROJECT_ROOT / os.getenv("DATA_PROCESSED_DIR", "data/processed")

# Se asegura la existencia física de los directorios de datos
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Se configuran las rutas específicas para MLOps y cumplimiento (modelos compartidos y ejecuciones)
SHARED_DIR = DATA_PROCESSED_DIR / "shared"
TRAIN_RUNS_DIR = DATA_PROCESSED_DIR / "train" / "runs"
USE_RUNS_DIR = DATA_PROCESSED_DIR / "use" / "runs"

SHARED_DIR.mkdir(parents=True, exist_ok=True)
(SHARED_DIR / "models").mkdir(parents=True, exist_ok=True)
TRAIN_RUNS_DIR.mkdir(parents=True, exist_ok=True)
USE_RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Se define el modo de ejecución por defecto ('use' o 'train')
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "use").strip().lower()

# Se genera un identificador único (RUN_ID) para garantizar trazabilidad y reproducibilidad
RUN_ID = os.getenv("RUN_ID", "").strip()
if not RUN_ID:
    RUN_ID = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.environ["RUN_ID"] = RUN_ID

RUN_DATE = RUN_ID

# Se asigna el directorio de salida correspondiente al modo de ejecución actual
if PIPELINE_MODE == "train":
    RUN_DIR = TRAIN_RUNS_DIR / f"run_{RUN_DATE}"
else:
    RUN_DIR = USE_RUNS_DIR / f"run_{RUN_DATE}"

RUN_DIR.mkdir(parents=True, exist_ok=True)

# Se configuran los parámetros de representación semántica (embeddings)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers").strip().lower()
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Se inicializa el servicio de registro de eventos (logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'pld_aml_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("AML_System")

logger.info(f"configuración: backend={EMBEDDING_BACKEND}")
