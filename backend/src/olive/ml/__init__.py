"""Olive local ML pipeline package.

Submodules pull in heavy third-party stacks (torch, pyannote-audio,
onnxruntime, transformers — seconds of import time), so they are loaded
lazily via PEP 562 module ``__getattr__``. In particular, importing
``olive.ml.model_store`` for container preflight must NOT pay for neural
dependencies. Direct submodule imports (``import olive.ml.diarize``) are
unaffected.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..schemas import coalesce_speaker_segments

# Attribute -> submodule that defines it. Kept in sync with the eager imports
# this file used to have.
_LAZY_ATTRS: dict[str, str] = {
    "CtcForcedAligner": "align",
    "proportional_word_timestamps": "align",
    "CohereAsrEngine": "asr_cohere",
    "MockLocalAsrEngine": "asr_cohere",
    "TransformersAsrEngine": "asr_cohere",
    "SPEECH_ENHANCEMENT_FILTER": "audio",
    "DecodedWav": "audio",
    "decode_wav": "audio",
    "encode_wav": "audio",
    "enhance_audio_file": "audio",
    "enhance_audio_file_async": "audio",
    "is_wav": "audio",
    "load_audio_samples": "audio",
    "resample": "audio",
    "decodeWav": "audio",
    "encodeWav": "audio",
    "isWav": "audio",
    "loadAudioSamples": "audio",
    "enhanceAudioFile": "audio",
    "enhanceAudioFileAsync": "audio",
    "LocalSpeakerDiarizer": "diarize",
    "PyannoteSpeakerDiarizer": "diarize",
    "SherpaSpeakerDiarizer": "diarize",
    "WESPEAKER_FILENAME": "embed",
    "WESPEAKER_MODEL_TAG": "embed",
    "WESPEAKER_REPO_ID": "embed",
    "AcousticFeatureEmbeddingExtractor": "embed",
    "SherpaSpeakerEmbeddingExtractor": "embed",
    "WeSpeakerEmbeddingExtractor": "embed",
    "identify_cloud_transcript_speakers": "identify",
    "identify_speakers_from_turns": "identify",
    "match_cluster_centroids": "identify",
    "rename_transcript_speakers": "identify",
    "turns_from_transcript": "identify",
    "LocalTranscriptionPipeline": "pipeline",
    "Aligner": "protocols",
    "AsrBackend": "protocols",
    "Diarizer": "protocols",
    "DiarizerInterface": "protocols",
    "DiscoveredSpeakerVoiceprint": "protocols",
    "LocalAsrEngineInterface": "protocols",
    "LocalSpeakerSegment": "protocols",
    "LocalTranscriptionResult": "protocols",
    "SpeakerEmbedder": "protocols",
    "SpeakerEmbeddingExtractorInterface": "protocols",
    "VoiceprintVector": "protocols",
    "cosine_similarity": "scoring",
    "cosineSimilarity": "scoring",
    "l2_norm": "scoring",
    "l2Norm": "scoring",
    "merge_voiceprint_vectors": "scoring",
    "merge_weighted_voiceprint_vectors": "scoring",
    "mergeVoiceprintVectors": "scoring",
    "mergeWeightedVoiceprintVectors": "scoring",
    "normalize_vector": "scoring",
    "normalizeVector": "scoring",
    "score_against_trusted_vectors": "scoring",
    "scoreAgainstTrustedVectors": "scoring",
    "update_voiceprint_centroid": "scoring",
    "updateVoiceprintCentroid": "scoring",
    "AbortController": "worker",
    "AbortSignal": "worker",
}


def __getattr__(name: str) -> Any:
    if name == "coalesceSpeakerSegments":
        return coalesce_speaker_segments
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "AbortController",
    "identify_cloud_transcript_speakers",
    "identify_speakers_from_turns",
    "match_cluster_centroids",
    "rename_transcript_speakers",
    "turns_from_transcript",
    "AbortSignal",
    "AcousticFeatureEmbeddingExtractor",
    "Aligner",
    "AsrBackend",
    "CohereAsrEngine",
    "CtcForcedAligner",
    "DecodedWav",
    "Diarizer",
    "DiarizerInterface",
    "DiscoveredSpeakerVoiceprint",
    "LocalAsrEngineInterface",
    "LocalSpeakerDiarizer",
    "LocalSpeakerSegment",
    "LocalTranscriptionPipeline",
    "LocalTranscriptionResult",
    "MockLocalAsrEngine",
    "PyannoteSpeakerDiarizer",
    "SPEECH_ENHANCEMENT_FILTER",
    "SherpaSpeakerDiarizer",
    "SherpaSpeakerEmbeddingExtractor",
    "SpeakerEmbedder",
    "SpeakerEmbeddingExtractorInterface",
    "TransformersAsrEngine",
    "VoiceprintVector",
    "WESPEAKER_FILENAME",
    "WESPEAKER_MODEL_TAG",
    "WESPEAKER_REPO_ID",
    "WeSpeakerEmbeddingExtractor",
    "coalesceSpeakerSegments",
    "coalesce_speaker_segments",
    "cosineSimilarity",
    "cosine_similarity",
    "decodeWav",
    "decode_wav",
    "encodeWav",
    "encode_wav",
    "enhanceAudioFile",
    "enhanceAudioFileAsync",
    "enhance_audio_file",
    "enhance_audio_file_async",
    "isWav",
    "is_wav",
    "l2Norm",
    "l2_norm",
    "loadAudioSamples",
    "load_audio_samples",
    "mergeVoiceprintVectors",
    "mergeWeightedVoiceprintVectors",
    "merge_voiceprint_vectors",
    "merge_weighted_voiceprint_vectors",
    "normalizeVector",
    "normalize_vector",
    "proportional_word_timestamps",
    "resample",
    "scoreAgainstTrustedVectors",
    "score_against_trusted_vectors",
    "updateVoiceprintCentroid",
    "update_voiceprint_centroid",
]
