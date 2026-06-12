"""
db.py — Supabase multi-tenant database layer for CortexOS Workspace
Handles explicit routing for both authenticated accounts and Guest access models.
"""

import os
import mimetypes
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL: str = os.environ["SUPABASE_URL"]

# Backend always uses the service role key so it can bypass RLS.
# NEVER expose this key in the frontend — it's only safe server-side.
SUPABASE_KEY: str = (
    os.environ.get("SUPABASE_SERVICE_KEY")
    or os.environ.get("SUPABASE_ANON_KEY")  # fallback so local dev still works
)

_supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# ──────────────────────────────────────────────────────────────
# Profile & Session Provisioning
# ──────────────────────────────────────────────────────────────

def ensure_profile_exists(user_id: str, display_name: str = "Guest User", email: str = None) -> dict:
    is_guest = (display_name == "Guest User")
    payload = {
        "id": user_id,
        "display_name": display_name,
        "email": email,
        "is_guest": is_guest,
        "updated_at": _now_iso()
    }
    result = (
        _supabase
        .table("profiles")
        .upsert(payload, on_conflict="id")
        .execute()
    )
    return result.data[0] if result.data else {}

# ──────────────────────────────────────────────────────────────
# Multi-Tenant Document Registry
# ──────────────────────────────────────────────────────────────

def register_user_document(user_id: str, filename: str, storage_path: str, file_size_meta: str) -> dict:
    ensure_profile_exists(user_id)
    payload = {
        "user_id": user_id,
        "filename": filename,
        "storage_path": storage_path,
        "file_size_meta": file_size_meta,
        "created_at": _now_iso()
    }
    result = (
        _supabase
        .table("user_documents")
        .insert(payload)
        .execute()
    )
    return result.data[0] if result.data else {}

def get_tenant_documents(user_id: str) -> list[dict]:
    result = (
        _supabase
        .table("user_documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []

# ──────────────────────────────────────────────────────────────
# Permanent Supabase Cloud Storage Engine
# ──────────────────────────────────────────────────────────────

def upload_file_to_cloud(user_id: str, file_bytes: bytes, filename: str) -> str:
    cloud_path = f"{user_id}/{filename}"

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"

    _supabase.storage.from_("documents").upload(
        path=cloud_path,
        file=file_bytes,
        file_options={
            "content-type": content_type,
            "upsert": "true",
        }
    )

    return cloud_path


def upload_file_from_path(user_id: str, local_path: str, filename: str) -> str:
    """
    Memory-friendly upload: hands Supabase the file path so the client reads
    from disk instead of us holding the whole file in RAM as bytes.
    """
    cloud_path = f"{user_id}/{filename}"

    content_type, _ = mimetypes.guess_type(filename)
    if not content_type:
        content_type = "application/octet-stream"

    _supabase.storage.from_("documents").upload(
        path=cloud_path,
        file=local_path,
        file_options={
            "content-type": content_type,
            "upsert": "true",
        }
    )

    return cloud_path