"""
Voice Modality — Audio Dataset Loader

Converts labeled audio recordings into standardized tabular datasets for ML model
training and evaluation.

Pipeline per audio sample:
    Raw Audio File -> Preprocessing (mono, 16 kHz, normalization, silence handling)
                   -> Combined Feature Extraction (41 features: 26 MFCC + 4 Pitch + 3 Energy + 8 Spectral)
                   -> Tabular Row: [participant_id, 41 voice features, target]

Adheres to:
    - Centralized config: src/ai_model/voice/config.py (FeatureConfig, AudioConfig, ModelConfig, PathConfig)
    - Input: metadata.csv / pd.DataFrame with audio filenames, participant IDs, and target labels
    - Output: pd.DataFrame with columns: [participant_id] + 41 feature columns + [target]
    - Strict validation: missing files, invalid labels, corrupt audio, and extraction errors
    - Preserves anonymous participant IDs and user-defined target labels without hardcoding.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from .config import (
    AUDIO_CONFIG,
    FEATURE_CONFIG,
    MODEL_CONFIG,
    PATH_CONFIG,
    AudioConfig,
    FeatureConfig,
    ModelConfig,
    PathConfig,
)
from .features import (
    COMBINED_FEATURE_NAMES,
    extract_combined_features,
    get_combined_feature_names,
)
from .preprocessing import PreprocessedAudio, preprocess_audio
from .recording import AudioError, extract_participant_id


# ===========================================================================
# Exceptions
# ===========================================================================

class DatasetError(AudioError):
    """Base exception raised for dataset loading and processing failures."""
    pass


class DatasetNotFoundError(DatasetError):
    """Raised when metadata file or audio directory cannot be found."""
    pass


class DatasetMetadataError(DatasetError):
    """Raised when metadata is empty, malformed, or missing required columns."""
    pass


class DatasetAudioNotFoundError(DatasetError):
    """Raised when an audio file referenced in metadata does not exist on disk."""
    pass


class DatasetLabelError(DatasetError):
    """Raised when a target label is missing, null, or invalid."""
    pass


class DatasetExtractionError(DatasetError):
    """Raised when audio preprocessing or feature extraction fails on an audio file."""
    pass


# ===========================================================================
# Dataset Loader Controller
# ===========================================================================

class AudioDatasetLoader:
    """
    Dataset loader for converting labeled audio recordings into tabular feature datasets.

    Extracts the full 41-feature acoustic representation for each audio file in the
    dataset metadata and returns a unified pandas DataFrame.
    """

    # Common column name aliases for automated column discovery
    _FILENAME_CANDIDATES: Tuple[str, ...] = (
        "filename", "file_name", "audio_file", "audio_path", "filepath", "file", "wav_file"
    )
    _PARTICIPANT_CANDIDATES: Tuple[str, ...] = (
        "participant_id", "participant", "subject_id", "subject", "user_id", "id"
    )
    _LABEL_CANDIDATES: Tuple[str, ...] = (
        "target", "label", "class", "risk_level", "stress_level", "category", "status"
    )

    def __init__(
        self,
        config: FeatureConfig = FEATURE_CONFIG,
        audio_config: AudioConfig = AUDIO_CONFIG,
        model_config: ModelConfig = MODEL_CONFIG,
        path_config: PathConfig = PATH_CONFIG,
        audio_dir: Optional[Union[str, Path]] = None,
        filename_column: Optional[str] = None,
        participant_id_column: Optional[str] = None,
        label_column: Optional[str] = None,
        on_error: str = "raise",
    ):
        """
        Initialize the AudioDatasetLoader.

        Parameters:
            config: FeatureConfig settings for acoustic feature extraction.
            audio_config: AudioConfig settings for preprocessing.
            model_config: ModelConfig settings for column conventions.
            path_config: PathConfig settings for directory paths.
            audio_dir: Root directory containing audio files. If None, defaults
                       to path_config.raw_audio_dir or directory of metadata file.
            filename_column: Explicit column name for audio filenames.
            participant_id_column: Explicit column name for participant IDs.
            label_column: Explicit column name for target labels.
            on_error: Error handling mode ('raise' to fail immediately on bad file/label,
                      'skip' to log warning and skip faulty samples).
        """
        self.config = config
        self.audio_config = audio_config
        self.model_config = model_config
        self.path_config = path_config
        self.audio_dir = Path(audio_dir) if audio_dir is not None else None
        self.filename_column = filename_column
        self.participant_id_column = (
            participant_id_column or model_config.participant_id_column
        )
        self.label_column = label_column or model_config.label_column

        if on_error not in ("raise", "skip"):
            raise ValueError(f"on_error must be 'raise' or 'skip', got '{on_error}'.")
        self.on_error = on_error

    def _resolve_columns(self, df: pd.DataFrame) -> Tuple[str, str, str]:
        """
        Identify filename, participant_id, and label column names from DataFrame.

        Returns:
            Tuple of (filename_col, participant_id_col, label_col).

        Raises:
            DatasetMetadataError: If required columns cannot be resolved.
        """
        cols = list(df.columns)

        # 1. Resolve filename column
        fn_col: Optional[str] = None
        if self.filename_column and self.filename_column in df.columns:
            fn_col = self.filename_column
        else:
            for candidate in self._FILENAME_CANDIDATES:
                if candidate in df.columns:
                    fn_col = candidate
                    break
        if fn_col is None:
            raise DatasetMetadataError(
                f"Cannot identify audio filename column in metadata. "
                f"Available columns: {cols}. Expected one of {list(self._FILENAME_CANDIDATES)} "
                "or specify 'filename_column' explicitly."
            )

        # 2. Resolve participant ID column
        pid_col: Optional[str] = None
        if self.participant_id_column and self.participant_id_column in df.columns:
            pid_col = self.participant_id_column
        else:
            for candidate in self._PARTICIPANT_CANDIDATES:
                if candidate in df.columns:
                    pid_col = candidate
                    break
        if pid_col is None:
            raise DatasetMetadataError(
                f"Cannot identify participant ID column in metadata. "
                f"Available columns: {cols}. Expected one of {list(self._PARTICIPANT_CANDIDATES)} "
                "or specify 'participant_id_column' explicitly."
            )

        # 3. Resolve target label column
        lbl_col: Optional[str] = None
        if self.label_column and self.label_column in df.columns:
            lbl_col = self.label_column
        else:
            for candidate in self._LABEL_CANDIDATES:
                if candidate in df.columns:
                    lbl_col = candidate
                    break
        if lbl_col is None:
            raise DatasetMetadataError(
                f"Cannot identify target label column in metadata. "
                f"Available columns: {cols}. Expected one of {list(self._LABEL_CANDIDATES)} "
                "or specify 'label_column' explicitly."
            )

        return fn_col, pid_col, lbl_col

    def _resolve_audio_path(
        self,
        raw_filename: str,
        audio_root: Optional[Path],
    ) -> Path:
        """
        Locate the audio file on disk, checking absolute and relative possibilities.

        Parameters:
            raw_filename: Filename or partial path from metadata.
            audio_root: Candidate base directory.

        Returns:
            Resolved absolute Path.

        Raises:
            DatasetAudioNotFoundError: If file does not exist.
        """
        candidate_path = Path(raw_filename)

        # If already an existing absolute path
        if candidate_path.is_absolute() and candidate_path.exists():
            return candidate_path

        # If relative to audio_root
        if audio_root is not None:
            joined = (audio_root / candidate_path).resolve()
            if joined.exists():
                return joined

        # If relative to PathConfig.raw_audio_dir
        config_path = (Path(self.path_config.raw_audio_dir) / candidate_path).resolve()
        if config_path.exists():
            return config_path

        # If relative to current working directory
        cwd_path = candidate_path.resolve()
        if cwd_path.exists():
            return cwd_path

        checked_dirs = [str(audio_root)] if audio_root else []
        checked_dirs.append(self.path_config.raw_audio_dir)
        raise DatasetAudioNotFoundError(
            f"Audio file '{raw_filename}' does not exist. Looked in: {checked_dirs}."
        )

    def load(
        self,
        metadata: Union[str, Path, pd.DataFrame],
        audio_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Load audio files referenced in metadata, run preprocessing + feature extraction,
        and return a structured pandas DataFrame.

        Parameters:
            metadata: Path to metadata CSV file or an existing pandas DataFrame.
            audio_dir: Directory containing audio files. If None, uses metadata file's
                       parent directory or self.audio_dir.

        Returns:
            pandas.DataFrame with columns:
                [participant_id_col] + 41 feature columns + [label_col]

        Raises:
            DatasetNotFoundError: If metadata file path is missing.
            DatasetMetadataError: If metadata is empty or missing columns.
            DatasetAudioNotFoundError: If an audio file cannot be found.
            DatasetLabelError: If a target label is null, NaN, or empty.
            DatasetExtractionError: If feature extraction fails on an audio file.
        """
        # 1. Load or validate metadata DataFrame
        metadata_source_dir: Optional[Path] = None
        if isinstance(metadata, (str, Path)):
            meta_path = Path(metadata)
            if not meta_path.exists():
                raise DatasetNotFoundError(
                    f"Metadata file does not exist: {meta_path.resolve()}"
                )
            try:
                df_meta = pd.read_csv(meta_path)
            except Exception as e:
                raise DatasetMetadataError(
                    f"Failed to parse metadata CSV at '{meta_path}': {e}"
                ) from e
            metadata_source_dir = meta_path.parent
        elif isinstance(metadata, pd.DataFrame):
            df_meta = metadata.copy()
        else:
            raise TypeError(
                f"metadata must be a file path (str/Path) or pandas.DataFrame, got {type(metadata)}."
            )

        if df_meta.empty:
            raise DatasetMetadataError("Metadata DataFrame is empty (0 rows).")

        # 2. Resolve columns
        fn_col, pid_col, lbl_col = self._resolve_columns(df_meta)

        # 3. Determine audio root directory
        effective_audio_dir: Optional[Path] = (
            Path(audio_dir)
            if audio_dir is not None
            else (self.audio_dir if self.audio_dir is not None else metadata_source_dir)
        )

        # 4. Feature names (41 features)
        feature_names = get_combined_feature_names(
            n_mfcc=self.config.n_mfcc,
            stats=self.config.aggregation_stats,
        )

        # 5. Extract features for each sample
        rows: List[Dict[str, Any]] = []

        for idx, row in df_meta.iterrows():
            raw_fn = row[fn_col]
            raw_pid = row[pid_col]
            raw_lbl = row[lbl_col]

            # Validate target label
            if pd.isna(raw_lbl) or raw_lbl is None or (isinstance(raw_lbl, str) and not raw_lbl.strip()):
                msg = (
                    f"Invalid target label '{raw_lbl}' at metadata row {idx} "
                    f"for audio file '{raw_fn}'."
                )
                if self.on_error == "raise":
                    raise DatasetLabelError(msg)
                warnings.warn(f"{msg} Skipping sample.")
                continue

            # Validate / format participant ID
            if pd.isna(raw_pid) or raw_pid is None or (isinstance(raw_pid, str) and not raw_pid.strip()):
                # Fallback: extract from filename if possible, otherwise raise
                extracted_id = extract_participant_id(str(raw_fn))
                if extracted_id:
                    pid_val = extracted_id
                else:
                    msg = (
                        f"Missing participant ID at metadata row {idx} for audio file '{raw_fn}'."
                    )
                    if self.on_error == "raise":
                        raise DatasetMetadataError(msg)
                    warnings.warn(f"{msg} Skipping sample.")
                    continue
            else:
                pid_val = str(raw_pid).strip()

            # Resolve audio file path
            try:
                audio_path = self._resolve_audio_path(str(raw_fn), effective_audio_dir)
            except DatasetAudioNotFoundError as err:
                if self.on_error == "raise":
                    raise err
                warnings.warn(f"{err} Skipping sample.")
                continue

            # Run Preprocessing + Combined Feature Extraction Pipeline
            try:
                preprocessed: PreprocessedAudio = preprocess_audio(
                    audio=audio_path,
                    config=self.audio_config,
                )
                feature_vec: np.ndarray = extract_combined_features(
                    audio=preprocessed,
                    config=self.config,
                )
            except Exception as err:
                msg = f"Feature extraction failed for file '{audio_path}' (row {idx}): {err}"
                if self.on_error == "raise":
                    raise DatasetExtractionError(msg) from err
                warnings.warn(f"{msg} Skipping sample.")
                continue

            # Assemble row with deterministic order: participant_id, 41 features, target
            sample_record: Dict[str, Any] = {pid_col: pid_val}
            for name, val in zip(feature_names, feature_vec):
                sample_record[name] = float(val)
            sample_record[lbl_col] = raw_lbl

            rows.append(sample_record)

        if len(rows) == 0:
            raise DatasetMetadataError(
                "No valid audio samples could be processed from the metadata."
            )

        # 6. Build and order DataFrame columns
        ordered_columns = [pid_col] + feature_names + [lbl_col]
        dataset_df = pd.DataFrame(rows, columns=ordered_columns)

        # Ensure feature columns are float32
        for col in feature_names:
            dataset_df[col] = dataset_df[col].astype(np.float32)

        return dataset_df

    def save(
        self,
        df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """
        Save the extracted dataset DataFrame to a CSV file.

        Parameters:
            df: Extracted dataset DataFrame.
            output_path: Target CSV file path (defaults to PathConfig.features_filename).

        Returns:
            Resolved Path of the saved CSV.
        """
        if output_path is None:
            out = Path(self.path_config.processed_dir) / self.path_config.features_filename
        else:
            out = Path(output_path)

        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return out.resolve()


# Alias for class name flexibility
VoiceDatasetLoader = AudioDatasetLoader


# ===========================================================================
# Functional API
# ===========================================================================

def load_voice_dataset(
    metadata: Union[str, Path, pd.DataFrame],
    audio_dir: Optional[Union[str, Path]] = None,
    config: FeatureConfig = FEATURE_CONFIG,
    audio_config: AudioConfig = AUDIO_CONFIG,
    model_config: ModelConfig = MODEL_CONFIG,
    path_config: PathConfig = PATH_CONFIG,
    filename_column: Optional[str] = None,
    participant_id_column: Optional[str] = None,
    label_column: Optional[str] = None,
    on_error: str = "raise",
) -> pd.DataFrame:
    """
    Load labeled voice recordings into a tabular 41-feature dataset DataFrame.

    Convenience wrapper around AudioDatasetLoader.load().

    Parameters:
        metadata: Path to metadata CSV or an existing DataFrame.
        audio_dir: Root directory containing audio files (optional).
        config: Acoustic FeatureConfig.
        audio_config: Preprocessing AudioConfig.
        model_config: Model configuration.
        path_config: Path configuration.
        filename_column: Metadata column name containing audio filenames.
        participant_id_column: Metadata column name containing participant IDs.
        label_column: Metadata column name containing target labels.
        on_error: 'raise' (default) or 'skip'.

    Returns:
        pandas.DataFrame with columns:
            [participant_id_col] + 41 voice features + [label_col]
    """
    loader = AudioDatasetLoader(
        config=config,
        audio_config=audio_config,
        model_config=model_config,
        path_config=path_config,
        audio_dir=audio_dir,
        filename_column=filename_column,
        participant_id_column=participant_id_column,
        label_column=label_column,
        on_error=on_error,
    )
    return loader.load(metadata=metadata, audio_dir=audio_dir)


# Alias
load_audio_dataset = load_voice_dataset


def save_voice_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Save the extracted voice dataset DataFrame to CSV.

    Parameters:
        df: Dataset DataFrame.
        output_path: Destination CSV file path.

    Returns:
        Resolved Path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out.resolve()
