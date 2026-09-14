"""
Voice Modality — Data Module

Re-exports dataset loading functionality from dataset.py.
"""

from .dataset import (
    AudioDatasetLoader,
    DatasetAudioNotFoundError,
    DatasetError,
    DatasetExtractionError,
    DatasetLabelError,
    DatasetMetadataError,
    DatasetNotFoundError,
    VoiceDatasetLoader,
    load_audio_dataset,
    load_voice_dataset,
    save_voice_dataset,
)

__all__ = [
    "AudioDatasetLoader",
    "VoiceDatasetLoader",
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetMetadataError",
    "DatasetAudioNotFoundError",
    "DatasetLabelError",
    "DatasetExtractionError",
    "load_voice_dataset",
    "load_audio_dataset",
    "save_voice_dataset",
]
