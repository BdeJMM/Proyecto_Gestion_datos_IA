import os
import oracledb
import sqlalchemy
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  

WALLET_DIR = Path(os.getenv("ORACLE_WALLET_DIR", "Wallet_GESTIONDATOIA")).resolve()
USUARIO    = os.getenv("ORACLE_USER", "ADMIN")
PASSWORD   = os.getenv("ORACLE_PASSWORD")
DSN        = os.getenv("ORACLE_DSN", "gestiondatosia_tp")
WALLET_PWD = os.getenv("ORACLE_WALLET_PASSWORD")

_engine = None

def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    def _creator():
        return oracledb.connect(
            user            = USUARIO,
            password        = PASSWORD,
            dsn             = DSN,
            config_dir      = str(WALLET_DIR),
            wallet_location = str(WALLET_DIR),
            wallet_password = WALLET_PWD,
        )

    _engine = sqlalchemy.create_engine(
        "oracle+oracledb://",
        creator      = _creator,
        pool_size    = 2,
        max_overflow = 0,
        pool_pre_ping= True,
    )
    return _engine