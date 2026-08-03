from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "09-智能风控与量化建模赛道-江苏银行-基于资金图谱的涉诈账户发现与可疑链路解释"

RAW_ACCOUNTS_PATH = DATA_DIR / "账户节点表.xlsx"
RAW_TRANSACTIONS_PATH = DATA_DIR / "交易边表.xlsx"
RAW_LABELS_PATH = DATA_DIR / "风险标签表.xlsx"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
DOCS_DIR = PROJECT_ROOT / "docs"
CLEAN_DIR = OUTPUT_DIR / "clean"
LABEL_DIR = OUTPUT_DIR / "labels"
FEATURE_DIR = OUTPUT_DIR / "features"
METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
EXPLANATION_DIR = OUTPUT_DIR / "explanations"
MODEL_DIR = PROJECT_ROOT / "models"
DELIVERABLE_DIR = PROJECT_ROOT / "deliverables"

RANDOM_SEED = 2026

TRAIN_START = "2025-07-01"
TRAIN_END = "2025-10-31 23:59:59"
VALID_START = "2025-11-01"
VALID_END = "2025-11-30 23:59:59"
TEST_START = "2025-12-01"
TEST_END = "2025-12-31 23:59:59"

LABEL_TEXT_TO_CODE = {
    "其它": 0,
    "嫌疑人": 1,
    "受害人": 2,
}

LABEL_CODE_TO_TEXT = {
    0: "其它",
    1: "嫌疑人",
    2: "受害人",
}

ID_COL = "account_id"
SRC_COL = "src_account_id"
DST_COL = "dst_account_id"
TIME_COL = "txn_time"
AMOUNT_COL = "amount"

SPLITS = {
    "train": (TRAIN_START, TRAIN_END),
    "valid": (VALID_START, VALID_END),
    "test": (TEST_START, TEST_END),
}
