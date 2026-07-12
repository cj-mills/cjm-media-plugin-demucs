"""Demucs v4 audio source separation capability — provides vocals extraction for removing background noise and music from speech audio.

Stage 8 (Option C / PILLAR 1c): the tool re-bases onto ToolCapability (pure
compute). The cache/persist bookends + the (input,config)->output_dir choice
moved OUT — the generic adapter (cjm-source-separation-adapter-interface) owns
the cache and tells the tool WHERE to write; the result DTO lives in
cjm-capability-primitives. Input is FULL-BAND audio (the model-ready convert
runs DOWNSTREAM of separation in the transcription pipeline)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import torch
from cjm_capability_primitives.source_separation import SourceSeparationResult
from cjm_substrate.core.capability import EnvVarSpec, RELOAD_TRIGGER, ToolCapability
from cjm_substrate.core.errors import CapabilityInputError
from cjm_substrate.utils.validation import (config_to_dict, dataclass_to_jsonschema, dict_to_config,
                                            SCHEMA_DESC, SCHEMA_ENUM, SCHEMA_MAX, SCHEMA_MIN,
                                            SCHEMA_TITLE)
from cjm_substrate_torch_utils.device import resolve_torch_device
from cjm_substrate_torch_utils.memory import release_model
from cjm_substrate_torch_utils.oom import cuda_oom_to_capability_resource_error


@dataclass
class DemucsCapabilityConfig:
    """Configuration for the Demucs processing capability."""
    
    model: str = field(
        default="htdemucs",
        metadata={
            SCHEMA_TITLE: "Model",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Demucs model to use for separation.",
            SCHEMA_ENUM: ["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"]
        }
    )
    
    device: str = field(
        default="auto",
        metadata={
            SCHEMA_TITLE: "Device",
            RELOAD_TRIGGER: "model",  # CR-4: change triggers model reload
            SCHEMA_DESC: "Compute device. 'auto' selects CUDA if available.",
            SCHEMA_ENUM: ["auto", "cpu", "cuda"]
        }
    )
    
    shifts: int = field(
        default=1,
        metadata={
            SCHEMA_TITLE: "Shifts",
            SCHEMA_DESC: "Number of random shifts for prediction averaging. Higher = better quality but slower.",
            SCHEMA_MIN: 0, SCHEMA_MAX: 10
        }
    )
    
    overlap: float = field(
        default=0.25,
        metadata={
            SCHEMA_TITLE: "Overlap",
            SCHEMA_DESC: "Overlap between prediction windows (0.0 to 0.5).",
            SCHEMA_MIN: 0.0, SCHEMA_MAX: 0.5
        }
    )
    
    segment: Optional[int] = field(
        default=None,
        metadata={
            SCHEMA_TITLE: "Segment Length",
            SCHEMA_DESC: "Split audio into segments of this many seconds for processing. None = process whole file."
        }
    )
    
    save_other_stems: bool = field(
        default=False,
        metadata={
            SCHEMA_TITLE: "Save Other Stems",
            SCHEMA_DESC: "Also save drums, bass, and other stems alongside vocals."
        }
    )
    
    output_format: str = field(
        default="wav",
        metadata={
            SCHEMA_TITLE: "Output Format",
            SCHEMA_DESC: "Format for output audio files.",
            SCHEMA_ENUM: ["wav", "flac", "mp3"]
        }
    )


class DemucsProcessingCapability(ToolCapability):
    """Demucs v4 source-separation tool capability for vocals extraction (stage 8: pure compute).

    Native-surface model (PILLAR 1c): this tool is PURE COMPUTE — `separate_vocals`
    loads the model, runs separation, and WRITES the vocals stem into the
    adapter-supplied `output_dir`, returning a typed `SourceSeparationResult`. The
    cache-check + the output-location choice + persistence live in the generic
    source-separation adapter (cjm-source-separation-adapter-interface); the result
    DTO lives in cjm-capability-primitives; identity is derived from the installed
    distribution. No `get_plugin_metadata`, no `self.storage`, no in-tool
    `cache_dir_for_config`. Input is FULL-BAND audio (the model-ready convert runs
    DOWNSTREAM of separation in the transcription pipeline)."""

    # CR-4: declarative reload-triggers — substrate's reconfigure_with_triggers
    # walks config_class for RELOAD_TRIGGER metadata and fires _release_<trigger>.
    config_class = DemucsCapabilityConfig

    # Track 19 (CR-12 worker-env model): worker spawn env declared on the class.
    # CUDA_VISIBLE_DEVICES is static; TORCH_HOME is templated to the substrate models
    # dir (Demucs downloads weights via torch.hub). The substrate injects at Popen.
    WORKER_ENV: ClassVar[List[EnvVarSpec]] = [
        EnvVarSpec(
            name="CUDA_VISIBLE_DEVICES",
            default="0",
            label="GPU Device",
            description="Which GPU index the worker uses.",
        ),
        EnvVarSpec(
            name="TORCH_HOME",
            default="${CJM_MODELS_DIR}/torch",
            label="Torch Hub Cache",
            description="torch.hub cache root for Demucs model downloads (templated to the substrate models dir).",
        ),
    ]

    def __init__(self):
        """Initialize the capability."""
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
        self.config: Optional[DemucsCapabilityConfig] = None
        self._separator = None  # Lazy-loaded Demucs Separator

    # ── Properties ──────────────────────────────────────────────────

    @property
    def name(self) -> str:  # Capability name identifier
        """Capability identity, derived from the installed distribution (PILLAR 1c)."""
        from importlib.metadata import metadata, packages_distributions
        dist = (packages_distributions().get(__package__) or [__package__.replace("_", "-")])[0]
        return metadata(dist)["Name"]

    @property
    def version(self) -> str:  # Capability version string
        """Get the capability version."""
        from cjm_capability_demucs import __version__
        return __version__

    # ── Lifecycle ────────────────────────────────────────────────────

    def initialize(self,
                  config: Optional[Any] = None,  # Configuration dict or None for defaults
                  ) -> None:
        """First-time setup. CR-4: config application factored into _apply_config; the
        substrate's reconfigure path fires _release_model on a model/device change then
        re-applies config. No storage init — the adapter owns the cache (stage 8)."""
        self._apply_config(config)
        self.logger.info(f"Initialized with model={self.config.model}, "
                         f"device={self.config.device}")

    def get_config_schema(self) -> Dict[str, Any]:  # JSON Schema for UI forms
        """Return JSON Schema for the capability configuration."""
        return dataclass_to_jsonschema(DemucsCapabilityConfig)

    def get_current_config(self) -> Dict[str, Any]:  # Current config as dict
        """Return the current configuration."""
        return config_to_dict(self.config) if self.config else {}

    def _apply_config(self,
                     config: Optional[Any] = None,  # Configuration dict or None for defaults
                     ) -> None:
        """CR-4: apply config values only. Called by initialize (first-time) and the
        substrate's reconfigure delta path. Model release on a model/device change is
        handled declaratively via RELOAD_TRIGGER -> _release_model (device resolved
        lazily in _load_model)."""
        self.config = dict_to_config(DemucsCapabilityConfig, config or {})

    def prefetch(self) -> None:
        """CR-4 (SG-19): eagerly load the model so the first execute() doesn't pay
        the download/load cost. Idempotent via _load_model's None-guard."""
        self._load_model()

    def on_disable(self) -> None:
        """CR-2: release the GPU model when the operator disables the capability (the
        worker stays alive); lazy reload on the next execute after re-enable."""
        self._release_model()

    def cleanup(self) -> None:
        """Clean up capability resources."""
        self._release_model()
        self.logger.info("Capability cleaned up")

    def is_available(self) -> bool:  # Whether the capability can run
        """Check if the capability is available on this system."""
        try:
            import demucs  # noqa
            return True
        except ImportError:
            return False

    def _load_model(self) -> None:
        """Load the Demucs Separator (lazy, cached).

        CR-4: a model/device change releases the separator declaratively via
        RELOAD_TRIGGER -> _release_model, so no manual change-detection is needed here —
        a None separator means a (re)load is required. The heartbeat wraps the WHOLE
        load: Separator() downloads weights via torch.hub on a cold cache (silent to the
        substrate's stall detector), so the heartbeat keeps the (progress, message)
        tuple advancing to avoid a false-positive stall."""
        if self._separator is not None:
            return

        device = resolve_torch_device(self.config.device)
        self.logger.info(f"Loading Demucs model '{self.config.model}' on {device}...")
        from demucs.api import Separator
        with self.heartbeat("loading Demucs model"):
            self._separator = Separator(
                model=self.config.model,
                device=device,
                shifts=self.config.shifts,
                overlap=self.config.overlap,
                segment=self.config.segment,
            )
        self.logger.info(f"Model loaded: samplerate={self._separator.samplerate}")

    def _release_model(self) -> None:
        """CR-4: release the Demucs Separator + free CUDA cache. RELOAD_TRIGGER target
        for model/device; on_disable / cleanup delegate here. Idempotent via
        cjm-substrate-torch-utils' release_model (no-op when already released)."""
        release_model(self, ["_separator"], device="cuda", logger=self.logger)

    def _prepare_audio(self,
                      audio: Union[str, Path]  # Path to the input audio file
                      ) -> str:  # The audio file path
        """Validate the audio input and return it as a path string (mirrors silero-vad).

        The adapter passes a file path; decode is the demucs Separator's job."""
        if isinstance(audio, (str, Path)):
            return str(audio)
        raise CapabilityInputError(  # SG-47: typed input-validation
            f"Unsupported audio input type: {type(audio)}; expected a file path (str or Path)",
            fields_invalid=["audio"],
        )

    def separate_vocals(self,
                        audio: Union[str, Path],  # Path to the input audio (full-band; NOT model-ready)
                        output_dir: str,          # Adapter-chosen dir to write the artifact into
                        **kwargs                  # Provenance pass-through (unused by compute)
                       ) -> SourceSeparationResult:  # Vocals-isolated artifact + metadata
        """Extract the vocals stem from an audio file — PURE COMPUTE.

        Stage 8 / PILLAR 1c: the cache-check + the (input,config)->output_dir choice +
        persistence moved to the generic adapter (cjm-source-separation-adapter-interface);
        this method loads the model, runs separation, WRITES the vocals stem (+ optional
        other stems) into the adapter-supplied `output_dir`, and returns the typed
        result. Separation params come from `self.config` (no per-call kwarg override —
        the tool runs its effective config). SG-47 Track B maps a CUDA OOM at the
        inference site to CapabilityResourceError (CR-7 reactive-retry reloads)."""
        input_p = Path(self._prepare_audio(audio))
        fmt = self.config.output_format
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load model (lazy, cached)
        self.report_progress(0.0, "Loading model...")
        self._load_model()

        # Run separation.
        self.report_progress(0.1, "Separating audio...")
        try:
            origin, separated = self._separator.separate_audio_file(input_p)
        except torch.cuda.OutOfMemoryError as e:
            raise cuda_oom_to_capability_resource_error(
                e, label=f"during Demucs separation (model={self.config.model!r})",
            ) from e

        # Save vocals into the adapter-supplied dir.
        self.report_progress(0.8, "Saving vocals...")
        from demucs.audio import save_audio
        vocals_path = out_dir / f"vocals.{fmt}"
        save_audio(separated["vocals"], str(vocals_path),
                   samplerate=self._separator.samplerate)

        # Optionally save other stems alongside.
        other_stems_saved = []
        if self.config.save_other_stems:
            for stem_name, stem_tensor in separated.items():
                if stem_name == "vocals":
                    continue
                stem_path = out_dir / f"{stem_name}.{fmt}"
                save_audio(stem_tensor, str(stem_path),
                           samplerate=self._separator.samplerate)
                other_stems_saved.append(str(stem_path))

        self.report_progress(1.0, "Complete")
        return SourceSeparationResult(
            output_path=str(vocals_path),
            metadata={
                "input_path": str(input_p),
                "duration": float(separated["vocals"].shape[-1]) / self._separator.samplerate,
                "sample_rate": self._separator.samplerate,
                "model": self.config.model,
                "stems_available": list(separated.keys()),
                "other_stems_saved": other_stems_saved,
                "device": str(self.config.device),
            },
        )
