import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load environment values from WrongNoteFlask/.env when present.
load_dotenv(ENV_PATH)


def env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        value = os.getenv(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def build_db_uri() -> str:
    direct_uri = env_first("DB_URI", default="")
    if direct_uri:
        return direct_uri

    driver = env_first("DB_DRIVER", default="ODBC Driver 18 for SQL Server")
    server = env_first("DB_SERVER", default="ms1901.gabiadb.com")
    database = env_first("DB_NAME", "DB_DATABASE", default="yujincast")
    username = env_first("DB_USER", "DB_USERNAME", default="")
    password = env_first("DB_PASSWORD", default="")

    odbc_connect = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=yes;"
    )
    return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc_connect)}"


class AppConfig:
    SQLALCHEMY_DATABASE_URI = build_db_uri()
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 1800,
        "pool_timeout": 30,
    }
    UPLOAD_FOLDER = str(BASE_DIR / "static" / "uploads")

    DB_SERVER = env_first("DB_SERVER", default="ms1901.gabiadb.com")
    DB_NAME = env_first("DB_NAME", "DB_DATABASE", default="yujincast")

    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5003"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
