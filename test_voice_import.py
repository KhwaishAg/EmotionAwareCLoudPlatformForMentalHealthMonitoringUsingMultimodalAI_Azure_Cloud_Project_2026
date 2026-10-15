"""Run this from the repo root to see the real voice import error:
    python test_voice_import.py
"""
try:
    from src.ai_model.voice.predict import predict_voice, VoiceModelNotTrainedError
    print("Voice module imported successfully.")
except Exception as error:
    import traceback
    print("VOICE IMPORT FAILED -- real error below:\n")
    traceback.print_exc()
