import os
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime


SUPPORTED_EXTENSIONS = (".csv", ".json", ".xlsx", ".xls", ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp")


def _load_manifest(path: str) -> Dict[str, Dict]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_manifest(path: str, data: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _is_supported(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTENSIONS)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def sync_s3(
    bucket: str,
    prefix: str,
    dest_dir: str,
    manifest_path: str,
    aws_profile: Optional[str] = None,
    aws_region: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    if not bucket:
        return [], []
    import boto3

    session_kwargs = {}
    if aws_profile:
        session_kwargs["profile_name"] = aws_profile
    session = boto3.Session(**session_kwargs) if session_kwargs else boto3.Session()
    client_kwargs = {}
    if aws_region:
        client_kwargs["region_name"] = aws_region
    s3 = session.client("s3", **client_kwargs)

    manifest = _load_manifest(manifest_path)
    downloaded = []
    meta_list = []

    continuation = None
    while True:
        list_kwargs = {"Bucket": bucket, "Prefix": prefix or ""}
        if continuation:
            list_kwargs["ContinuationToken"] = continuation
        resp = s3.list_objects_v2(**list_kwargs)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or not _is_supported(key):
                continue
            etag = obj.get("ETag", "").strip('"')
            last_modified = obj.get("LastModified")
            last_modified_iso = last_modified.isoformat() if last_modified else None
            size = obj.get("Size")
            prev = manifest.get(key)
            if prev and prev.get("etag") == etag:
                continue

            local_path = os.path.join(dest_dir, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            downloaded.append(local_path)

            manifest[key] = {
                "etag": etag,
                "last_modified": last_modified_iso,
                "size": size,
                "synced_at": _now_iso(),
                "path": local_path,
            }
            meta_list.append({
                "source": "s3",
                "bucket": bucket,
                "key": key,
                "path": local_path,
                "last_modified": last_modified_iso,
                "size": size,
            })

        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break

    _save_manifest(manifest_path, manifest)
    return downloaded, meta_list


def _adls_client(
    account_url: str,
    sas_token: Optional[str] = None,
    connection_string: Optional[str] = None,
    account_key: Optional[str] = None,
):
    from azure.storage.filedatalake import DataLakeServiceClient

    if connection_string:
        return DataLakeServiceClient.from_connection_string(connection_string)

    if sas_token:
        return DataLakeServiceClient(account_url=account_url, credential=sas_token)

    if account_key:
        return DataLakeServiceClient(account_url=account_url, credential=account_key)

    raise ValueError("ADLS requires a connection string, SAS token, or account key.")


def sync_adls(
    account_url: str,
    file_system: str,
    prefix: str,
    dest_dir: str,
    manifest_path: str,
    sas_token: Optional[str] = None,
    connection_string: Optional[str] = None,
    account_key: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    if not account_url or not file_system:
        return [], []
    client = _adls_client(
        account_url=account_url,
        sas_token=sas_token,
        connection_string=connection_string,
        account_key=account_key,
    )
    fs_client = client.get_file_system_client(file_system)

    manifest = _load_manifest(manifest_path)
    downloaded = []
    meta_list = []

    paths = fs_client.get_paths(path=prefix or "")
    for path in paths:
        if getattr(path, "is_directory", False):
            continue
        name = path.name
        if not _is_supported(name):
            continue
        etag = getattr(path, "etag", None)
        last_modified = getattr(path, "last_modified", None)
        last_modified_iso = last_modified.isoformat() if last_modified else None
        size = getattr(path, "content_length", None)
        prev = manifest.get(name)
        if prev and prev.get("etag") == etag:
            continue

        local_path = os.path.join(dest_dir, name)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        file_client = fs_client.get_file_client(name)
        data = file_client.download_file().readall()
        with open(local_path, "wb") as f:
            f.write(data)
        downloaded.append(local_path)

        manifest[name] = {
            "etag": etag,
            "last_modified": last_modified_iso,
            "size": size,
            "synced_at": _now_iso(),
            "path": local_path,
        }
        meta_list.append({
            "source": "adls",
            "file_system": file_system,
            "path": local_path,
            "remote_path": name,
            "last_modified": last_modified_iso,
            "size": size,
        })

    _save_manifest(manifest_path, manifest)
    return downloaded, meta_list
