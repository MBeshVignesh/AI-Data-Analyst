import os
import re
from dotenv import load_dotenv
from cloud_sync import sync_adls

load_dotenv()

CLOUD_INGEST_DIR = "./data/cloud_ingest"

def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value or "").strip("_")

def _normalize_adls_account_url(url: str) -> str:
    if not url:
        return url
    url = url.strip().rstrip("/")
    return url.replace(".blob.core.windows.net", ".dfs.core.windows.net")

auth_params = {}
if os.getenv("ADLS_SAS_TOKEN"):
    auth_params["sas_token"] = os.getenv("ADLS_SAS_TOKEN")
elif os.getenv("ADLS_ACCOUNT_KEY"):
    auth_params["account_key"] = os.getenv("ADLS_ACCOUNT_KEY")
elif os.getenv("ADLS_CONNECTION_STRING"):
    auth_params["connection_string"] = os.getenv("ADLS_CONNECTION_STRING")

downloaded, metadata = sync_adls(
    account_url=_normalize_adls_account_url(os.getenv("ADLS_ACCOUNT_URL")),
    file_system=os.getenv("ADLS_FILE_SYSTEM"),
    prefix=os.getenv("ADLS_PREFIX"),
    dest_dir=os.path.join(CLOUD_INGEST_DIR, "adls", _safe_name(os.getenv("ADLS_FILE_SYSTEM"))),
    manifest_path=os.path.join(CLOUD_INGEST_DIR, "_manifests", f"adls_{_safe_name(os.getenv('ADLS_FILE_SYSTEM'))}.json"),
    **auth_params
)

print("Downloaded files:", downloaded)
print("Metadata:", metadata)