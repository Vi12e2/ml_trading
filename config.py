import os, logging, warnings, optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    force=True
)
logger = logging.getLogger("StrategyTester")

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import swifter
    apply_fn = lambda ser, fn: ser.swifter.apply(fn)
except ImportError:
    apply_fn = lambda ser, fn: ser.apply(fn)

# Загрузка данных
crypto_tk = ["BTC-USD", "ETH-USD"]
stock_tk  = ["GOOGL"]
index_tk  = ["^IXIC"]

START_H, END_H = "2023-07-01", "2025-06-01"  # часовые данные
START_D        = "2022-01-01"                # дневные данные

# Сохранение
RAW_DIR        = "data"          # parquet + CSV
DATA_LAKE_DIR  = "data_lake"     # папка для признаков
DRIVE_ROOT     = "/content/drive/MyDrive/2325_alltk"
DRIVE_RAW      = os.path.join(DRIVE_ROOT, "raw_data")
DRIVE_PARQ     = os.path.join(DRIVE_ROOT, "trading.parquet")
DRIVE_LAKE     = os.path.join(DRIVE_ROOT, "data_lake")

# Параметры FE и горизонты
H        = 35      # целевой горизонт (часов)
SEQ_LEN  = 128     # длина окна
ATR_W    = 35      # ATR окно
ATR_DYN  = 65     # ATR динамическое окно

os.makedirs(RAW_DIR,     exist_ok=True)
os.makedirs(DATA_LAKE_DIR, exist_ok=True)

# Загрузка с Google Drive
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

    if os.path.exists(DRIVE_PARQ):
        RAW_DIR = DRIVE_ROOT
        logger.info("Чтение из %s", DRIVE_PARQ)
    elif os.path.isdir(DRIVE_RAW):
        RAW_DIR = DRIVE_RAW
        logger.info("Чтение из %s", DRIVE_RAW)
    else:
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info("Локальный RAW_DIR: %s", RAW_DIR)

    if os.path.isdir(DRIVE_LAKE):
        DATA_LAKE_DIR = DRIVE_LAKE
        logger.info("Сохранение data lake в %s", DRIVE_LAKE)
    else:
        os.makedirs(DATA_LAKE_DIR, exist_ok=True)
        logger.info("Локальный DATA_LAKE_DIR: %s", DATA_LAKE_DIR)

except Exception as e:
    logger.warning("Не удалось подключить Google Drive: %s", e)
    os.makedirs(RAW_DIR,       exist_ok=True)
    os.makedirs(DATA_LAKE_DIR, exist_ok=True)
