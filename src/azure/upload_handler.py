"""
Azure Cloud Platform — Secure Participant Audio Upload Handler

Implements secure, validated, participant-scoped audio upload operations
on top of the existing AudioBlobStorage client (Commit 19).

Guarantees:
    - Strict validation of participant IDs (anonymous, alphanumeric, no traversal).
    - Audio format and header inspection (WAV, MP3, FLAC, OGG, M4A, AAC, AIFF).
    - Safe deterministic blob paths: e.g. audio/<participant_id>/<filename>.
    - Accidental overwrite protection enabled by default.
    - Zero exposure of Azure credentials, tokens, or sensitive metadata.
    - Full reuse of AudioBlobStorage without code duplication.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import io
import mimetypes
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Set, Tuple, Union
from urllib.parse import urlparse, urlunparse

from .exceptions import (
    AzureStorageError,
    StorageFileNotFoundError,
    StorageUploadError,
    StorageValidationError,
)
from .storage import (
    AudioBlobStorage,
    validate_blob_name,
    validate_local_path_for_upload,
)


# ===========================================================================
# Supported Formats and Constants
# ===========================================================================

SUPPORTED_AUDIO_EXTENSIONS: Set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}

SUPPORTED_AUDIO_MIME_TYPES: Set[str] = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/vorbis",
    "audio/mp4",
    "audio/x-m4a",
    "audio/aac",
    "audio/aiff",
    "audio/x-aiff",
}

# Regex to enforce anonymous, alphanumeric participant IDs (with hyphens/underscores)
PARTICIPANT_ID_PATTERN: re.Pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

# Sensitive keywords blocked from custom blob metadata
SENSITIVE_METADATA_TERMS: Set[str] = {
    "key",
    "secret",
    "password",
    "token",
    "credential",
    "conn",
    "connection",
    "connection_string",
    "auth",
    "sas",
    "sig",
    "api_key",
}

DEFAULT_AUDIO_PREFIX: str = "audio"


# ===========================================================================
# Validation Utilities
# ===========================================================================

def validate_participant_id(participant_id: str) -> str:
    """
    Validate that participant_id is an anonymous, safe alphanumeric identifier.

    Rules:
        - Must be a non-empty string.
        - Must only contain letters, numbers, hyphens, and underscores (^[a-zA-Z0-9_-]+$).
        - Cannot contain path separators (/ or \\) or path traversal (..).
        - Maximum length of 128 characters.

    Parameters:
        participant_id: Candidate participant identifier.

    Returns:
        The validated participant ID string.

    Raises:
        StorageValidationError: If the participant ID fails validation.
    """
    if not isinstance(participant_id, str):
        raise StorageValidationError(
            f"Participant ID must be a string, got {type(participant_id).__name__}."
        )

    cleaned = participant_id.strip()
    if not cleaned:
        raise StorageValidationError("Participant ID cannot be empty or whitespace.")

    if len(cleaned) > 128:
        raise StorageValidationError(
            f"Participant ID exceeds maximum length of 128 characters (got {len(cleaned)})."
        )

    if not PARTICIPANT_ID_PATTERN.match(cleaned):
        raise StorageValidationError(
            f"Participant ID '{participant_id}' is invalid. "
            "It must be an anonymous identifier containing only alphanumeric characters, "
            "hyphens, or underscores, with no path separators, spaces, or special characters."
        )

    return cleaned


def sanitize_audio_filename(
    filename: Union[str, Path],
    allowed_extensions: Optional[Set[str]] = None,
) -> str:
    """
    Sanitize and validate an audio filename.

    Rules:
        - Disallows path separators (/ or \\) and path traversal (..).
        - Must have a supported audio extension (e.g. .wav, .mp3, .flac).
        - Disallows empty base names, hidden files, or null bytes.
        - Characters must be safe for cloud blob naming.

    Parameters:
        filename: Candidate filename or relative filename.
        allowed_extensions: Optional set of allowed extensions (defaults to SUPPORTED_AUDIO_EXTENSIONS).

    Returns:
        The sanitized filename.

    Raises:
        StorageValidationError: If filename contains path traversal or unsafe characters.
    """
    if filename is None:
        raise StorageValidationError("Filename cannot be None.")

    raw_name = str(filename).strip()
    if not raw_name:
        raise StorageValidationError("Filename cannot be empty or whitespace.")

    if "\x00" in raw_name:
        raise StorageValidationError("Filename cannot contain null bytes.")

    # Guard against explicit path separators and traversal attempts
    if "/" in raw_name or "\\" in raw_name:
        raise StorageValidationError(
            f"Filename must be a single file name, not a path with separators: '{raw_name}'."
        )

    if ".." in raw_name:
        raise StorageValidationError(
            f"Path traversal ('..') is strictly forbidden in filename: '{raw_name}'."
        )

    if raw_name.startswith("."):
        raise StorageValidationError(
            f"Invalid audio filename '{raw_name}': stem cannot be empty or hidden."
        )

    path = Path(raw_name)
    suffix = path.suffix.lower()
    allowed = allowed_extensions or SUPPORTED_AUDIO_EXTENSIONS

    if not suffix or suffix not in allowed:
        raise StorageValidationError(
            f"Unsupported audio file extension '{suffix}' for filename '{raw_name}'. "
            f"Supported extensions: {sorted(allowed)}."
        )

    stem = path.stem
    if not stem or stem.startswith("."):
        raise StorageValidationError(f"Invalid audio filename '{raw_name}': stem cannot be empty or hidden.")

    # Check for unsafe characters in filename
    if not re.match(r"^[a-zA-Z0-9_.-]+$", raw_name):
        raise StorageValidationError(
            f"Filename '{raw_name}' contains illegal characters. Only alphanumeric, "
            "underscores, hyphens, and dots are permitted."
        )

    return raw_name


def generate_audio_blob_path(
    participant_id: str,
    filename: Union[str, Path],
    prefix: str = DEFAULT_AUDIO_PREFIX,
) -> str:
    """
    Generate a safe, deterministic Azure Blob storage path for participant audio.

    Pattern:
        <prefix>/<participant_id>/<sanitized_filename>
        Example: audio/participant_001/recording_01.wav

    Parameters:
        participant_id: Validated anonymous participant identifier.
        filename: Sanitized audio filename.
        prefix: Optional root path prefix (default: 'audio').

    Returns:
        Deterministic blob path string.

    Raises:
        StorageValidationError: If participant ID, filename, or generated path is invalid.
    """
    valid_pid = validate_participant_id(participant_id)
    safe_filename = sanitize_audio_filename(filename)

    clean_prefix = prefix.replace("\\", "/").strip("/")
    if ".." in clean_prefix or clean_prefix.startswith("/"):
        raise StorageValidationError(f"Invalid path prefix: '{prefix}'.")

    blob_path = f"{clean_prefix}/{valid_pid}/{safe_filename}" if clean_prefix else f"{valid_pid}/{safe_filename}"
    return validate_blob_name(blob_path)


def detect_audio_format_from_bytes(data: bytes) -> Optional[str]:
    """
    Detect the audio container/format from raw header magic bytes.

    Supports:
        - WAV: 'RIFF....WAVE'
        - FLAC: 'fLaC'
        - OGG: 'OggS'
        - MP3: 'ID3' or sync frame (0xFF 0xFB/F3/F2)
        - AIFF: 'FORM....AIFF' or 'FORM....AIFC'
        - M4A: '....ftyp'

    Parameters:
        data: Raw byte buffer (at least first 12-16 bytes).

    Returns:
        Format string ('wav', 'flac', 'ogg', 'mp3', 'aiff', 'm4a') or None.
    """
    if len(data) < 4:
        return None

    # WAV check
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
        return "wav"

    # FLAC check
    if data[:4] == b"fLaC":
        return "flac"

    # OGG check
    if data[:4] == b"OggS":
        return "ogg"

    # MP3 check (ID3v2 or MPEG sync word)
    if data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"

    # AIFF check
    if data[:4] == b"FORM" and len(data) >= 12 and data[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"

    # M4A / MP4 container check
    if len(data) >= 8 and data[4:8] == b"ftyp":
        return "m4a"

    return None


def validate_audio_bytes(
    data: bytes,
    filename: Optional[str] = None,
    verify_magic_bytes: bool = True,
) -> bytes:
    """
    Validate an in-memory audio byte buffer before upload.

    Parameters:
        data: Raw audio bytes.
        filename: Optional filename to verify extension compatibility.
        verify_magic_bytes: Whether to verify audio header signatures.

    Returns:
        The validated raw bytes.

    Raises:
        StorageValidationError: If buffer is empty, wrong type, or has invalid audio header.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise StorageValidationError(
            f"Audio data must be bytes or bytearray, got {type(data).__name__}."
        )

    byte_data = bytes(data)
    if len(byte_data) == 0:
        raise StorageValidationError("Audio byte buffer cannot be empty (0 bytes).")

    if filename is not None:
        sanitize_audio_filename(filename)

    if verify_magic_bytes and len(byte_data) >= 4:
        detected_fmt = detect_audio_format_from_bytes(byte_data)
        if detected_fmt is None and filename is not None:
            ext = Path(filename).suffix.lower()
            # If standard audio format magic bytes do not match, reject invalid/corrupted payloads
            raise StorageValidationError(
                f"Data does not contain a valid audio header for '{filename}'. "
                "Expected valid WAV, MP3, FLAC, OGG, M4A, or AIFF audio stream."
            )

    return byte_data


def validate_audio_file(
    file_path: Union[str, Path],
    verify_magic_bytes: bool = True,
) -> Path:
    """
    Validate a local audio file before upload.

    Parameters:
        file_path: Local path to the candidate audio file.
        verify_magic_bytes: Whether to verify file magic bytes.

    Returns:
        Resolved Path object.

    Raises:
        StorageFileNotFoundError: If the local file does not exist.
        StorageValidationError: If file is empty, has unsupported extension, or invalid audio header.
    """
    path = validate_local_path_for_upload(file_path)

    # Check extension
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise StorageValidationError(
            f"Unsupported audio file extension '{suffix}' for file '{path.name}'. "
            f"Supported extensions: {sorted(SUPPORTED_AUDIO_EXTENSIONS)}."
        )

    # Check file size
    if path.stat().st_size == 0:
        raise StorageValidationError(f"Audio file '{path.name}' is empty (0 bytes).")

    # Inspect magic bytes from file header
    if verify_magic_bytes:
        with open(path, "rb") as f:
            header = f.read(32)
        if len(header) >= 4:
            detected_fmt = detect_audio_format_from_bytes(header)
            if detected_fmt is None:
                raise StorageValidationError(
                    f"File '{path.name}' does not contain a valid audio header. "
                    "Expected valid WAV, MP3, FLAC, OGG, M4A, or AIFF audio stream."
                )

    return path


def sanitize_blob_url(raw_url: str) -> str:
    """
    Sanitize an Azure Blob URL to prevent exposing query string credentials or SAS tokens.

    Parameters:
        raw_url: Candidate URL string.

    Returns:
        Clean canonical URL stripped of query parameters and authentication tokens.
    """
    if not raw_url:
        return ""
    parsed = urlparse(raw_url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def sanitize_upload_metadata(
    metadata: Optional[Dict[str, str]],
    participant_id: str,
    original_filename: str,
) -> Dict[str, str]:
    """
    Sanitize and construct safe metadata for Azure Blob Storage.

    Rules:
        - Injects non-sensitive participant_id and upload timestamp.
        - Validates that caller metadata does not expose credentials or secrets.
        - All values must be strings.

    Parameters:
        metadata: Optional user-supplied metadata dictionary.
        participant_id: Validated anonymous participant ID.
        original_filename: Validated base audio filename.

    Returns:
        Safe metadata dictionary.

    Raises:
        StorageValidationError: If metadata contains prohibited sensitive keys.
    """
    safe_metadata: Dict[str, str] = {
        "participant_id": participant_id,
        "original_filename": original_filename,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    if metadata:
        for k, v in metadata.items():
            k_clean = str(k).strip()
            k_lower = k_clean.lower()

            # Check against sensitive keywords
            if any(term in k_lower for term in SENSITIVE_METADATA_TERMS):
                raise StorageValidationError(
                    f"Metadata key '{k_clean}' contains sensitive term and cannot be attached to blob."
                )

            safe_metadata[k_clean] = str(v)

    return safe_metadata


# ===========================================================================
# Structured Upload Result
# ===========================================================================

@dataclass(frozen=True)
class AudioUploadResult:
    """
    Structured, secure result returned upon successful audio upload.

    Contains only non-sensitive operational metadata. Never exposes
    Azure connection strings, account keys, or raw security tokens.
    """
    blob_name: str
    container_name: str
    participant_id: str
    content_type: str
    size_bytes: int
    blob_url: str
    metadata: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Return safe dictionary representation."""
        return {
            "blob_name": self.blob_name,
            "container_name": self.container_name,
            "participant_id": self.participant_id,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "blob_url": self.blob_url,
            "metadata": dict(self.metadata),
        }


# ===========================================================================
# Secure Audio Upload Handler
# ===========================================================================

class SecureAudioUploadHandler:
    """
    High-level secure audio upload handler.

    Builds on top of AudioBlobStorage (Commit 19), providing:
        - Participant ID validation and scoping.
        - Strict audio format and header checks.
        - Safe deterministic path generation (audio/<participant_id>/<filename>).
        - Accidental overwrite protection by default.
        - Redaction/sanitization of sensitive credentials and metadata.
    """

    def __init__(
        self,
        storage: Optional[AudioBlobStorage] = None,
        default_prefix: str = DEFAULT_AUDIO_PREFIX,
    ):
        """
        Initialize SecureAudioUploadHandler.

        Parameters:
            storage: AudioBlobStorage client instance. If None, initializes from env.
            default_prefix: Root blob path prefix (default: 'audio').
        """
        self.storage = storage if storage is not None else AudioBlobStorage()
        self.default_prefix = default_prefix

    def upload_file(
        self,
        participant_id: str,
        file_path: Union[str, Path],
        filename: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        verify_magic_bytes: bool = True,
    ) -> AudioUploadResult:
        """
        Securely upload a local audio file for a validated participant.

        Parameters:
            participant_id: Anonymous participant identifier.
            file_path: Local path to the audio file.
            filename: Optional target filename. Defaults to file_path.name.
            overwrite: Whether to allow overwriting an existing blob (default: False).
            metadata: Optional safe key-value metadata.
            prefix: Optional path prefix (defaults to self.default_prefix).
            verify_magic_bytes: Whether to verify audio header bytes.

        Returns:
            AudioUploadResult containing safe upload details.

        Raises:
            StorageValidationError: On invalid participant ID, path, or audio format.
            StorageFileNotFoundError: If the local file does not exist.
            StorageUploadError: If upload fails or if blob exists and overwrite=False.
        """
        # Validate local audio file and participant
        valid_pid = validate_participant_id(participant_id)
        valid_path = validate_audio_file(file_path, verify_magic_bytes=verify_magic_bytes)

        target_filename = sanitize_audio_filename(filename or valid_path.name)
        active_prefix = prefix if prefix is not None else self.default_prefix
        blob_path = generate_audio_blob_path(valid_pid, target_filename, prefix=active_prefix)

        # Accidental overwrite protection
        if not overwrite and self.storage.blob_exists(blob_path):
            raise StorageUploadError(
                f"Blob '{blob_path}' already exists in container '{self.storage.container_name}' "
                "and overwrite is False."
            )

        # Build safe metadata
        safe_meta = sanitize_upload_metadata(
            metadata=metadata,
            participant_id=valid_pid,
            original_filename=valid_path.name,
        )

        # Resolve content type safely
        content_type = (
            mimetypes.guess_type(target_filename)[0]
            or ("audio/wav" if target_filename.endswith(".wav") else "application/octet-stream")
        )

        # Upload via existing AudioBlobStorage client (Commit 19)
        self.storage.upload_file(
            local_path=valid_path,
            blob_name=blob_path,
            overwrite=overwrite,
            content_type=content_type,
            metadata=safe_meta,
        )

        size_bytes = valid_path.stat().st_size
        raw_url = self.storage.get_blob_url(blob_path)
        safe_url = sanitize_blob_url(raw_url)

        return AudioUploadResult(
            blob_name=blob_path,
            container_name=self.storage.container_name,
            participant_id=valid_pid,
            content_type=content_type,
            size_bytes=size_bytes,
            blob_url=safe_url,
            metadata=safe_meta,
        )

    def upload_bytes(
        self,
        participant_id: str,
        data: bytes,
        filename: str,
        overwrite: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        verify_magic_bytes: bool = True,
    ) -> AudioUploadResult:
        """
        Securely upload raw audio bytes for a validated participant.

        Parameters:
            participant_id: Anonymous participant identifier.
            data: Audio byte buffer.
            filename: Target filename (e.g. 'recording_01.wav').
            overwrite: Whether to allow overwriting an existing blob (default: False).
            metadata: Optional safe key-value metadata.
            prefix: Optional path prefix (defaults to self.default_prefix).
            verify_magic_bytes: Whether to verify audio header bytes.

        Returns:
            AudioUploadResult containing safe upload details.

        Raises:
            StorageValidationError: On invalid participant ID, filename, or audio format.
            StorageUploadError: If upload fails or if blob exists and overwrite=False.
        """
        valid_pid = validate_participant_id(participant_id)
        target_filename = sanitize_audio_filename(filename)
        valid_data = validate_audio_bytes(
            data=data,
            filename=target_filename,
            verify_magic_bytes=verify_magic_bytes,
        )

        active_prefix = prefix if prefix is not None else self.default_prefix
        blob_path = generate_audio_blob_path(valid_pid, target_filename, prefix=active_prefix)

        # Accidental overwrite protection
        if not overwrite and self.storage.blob_exists(blob_path):
            raise StorageUploadError(
                f"Blob '{blob_path}' already exists in container '{self.storage.container_name}' "
                "and overwrite is False."
            )

        # Build safe metadata
        safe_meta = sanitize_upload_metadata(
            metadata=metadata,
            participant_id=valid_pid,
            original_filename=target_filename,
        )

        # Content type resolution
        content_type = (
            mimetypes.guess_type(target_filename)[0]
            or ("audio/wav" if target_filename.endswith(".wav") else "application/octet-stream")
        )

        # Upload via existing AudioBlobStorage client (Commit 19)
        self.storage.upload_bytes(
            data=valid_data,
            blob_name=blob_path,
            overwrite=overwrite,
            content_type=content_type,
            metadata=safe_meta,
        )

        raw_url = self.storage.get_blob_url(blob_path)
        safe_url = sanitize_blob_url(raw_url)

        return AudioUploadResult(
            blob_name=blob_path,
            container_name=self.storage.container_name,
            participant_id=valid_pid,
            content_type=content_type,
            size_bytes=len(valid_data),
            blob_url=safe_url,
            metadata=safe_meta,
        )

    def upload(
        self,
        participant_id: str,
        audio_source: Union[str, Path, bytes, bytearray],
        filename: Optional[str] = None,
        overwrite: bool = False,
        metadata: Optional[Dict[str, str]] = None,
        prefix: Optional[str] = None,
        verify_magic_bytes: bool = True,
    ) -> AudioUploadResult:
        """
        Convenience dispatcher to securely upload either a local file or byte buffer.

        Parameters:
            participant_id: Anonymous participant identifier.
            audio_source: Local file path (str or Path) or in-memory audio bytes.
            filename: Target filename (required if audio_source is bytes).
            overwrite: Whether to overwrite existing blob (default: False).
            metadata: Optional safe key-value metadata.
            prefix: Optional path prefix.
            verify_magic_bytes: Whether to verify audio header bytes.

        Returns:
            AudioUploadResult containing safe upload details.
        """
        if isinstance(audio_source, (bytes, bytearray)):
            if not filename:
                raise StorageValidationError(
                    "Filename must be provided when uploading raw audio bytes."
                )
            return self.upload_bytes(
                participant_id=participant_id,
                data=bytes(audio_source),
                filename=filename,
                overwrite=overwrite,
                metadata=metadata,
                prefix=prefix,
                verify_magic_bytes=verify_magic_bytes,
            )
        elif isinstance(audio_source, (str, Path)):
            return self.upload_file(
                participant_id=participant_id,
                file_path=audio_source,
                filename=filename,
                overwrite=overwrite,
                metadata=metadata,
                prefix=prefix,
                verify_magic_bytes=verify_magic_bytes,
            )
        else:
            raise StorageValidationError(
                f"Unsupported audio source type: {type(audio_source).__name__}. "
                "Expected file path (str, Path) or audio bytes (bytes, bytearray)."
            )


# Aliases for flexible import conventions
SecureAudioUploader = SecureAudioUploadHandler
ParticipantAudioUploader = SecureAudioUploadHandler
