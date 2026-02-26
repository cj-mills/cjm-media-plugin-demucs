"""Direct test script for the Demucs processing plugin.

Run from the repo root:
    python tests_manual/test_demucs_plugin.py

Tests:
1. Import and metadata generation
2. Plugin instantiation and config schema
3. Initialize and cleanup lifecycle
4. is_available check
5. Vocals separation on test audio
6. Verify job stored in database
"""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Test audio — reuse the fork's test file
TEST_FILE = Path("/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills/cjm-demucs-v4/test_files/segment_000.mp3")


def test_import_and_metadata():
    """Test imports and metadata generation."""
    print("=" * 60)
    print("Test 1: Import and metadata")
    print("=" * 60)
    from cjm_media_plugin_demucs.meta import get_plugin_metadata

    metadata = get_plugin_metadata()
    print(json.dumps(metadata, indent=2))
    assert metadata["name"] == "cjm-media-plugin-demucs"
    assert metadata["type"] == "media-processing"
    assert metadata["resources"]["requires_gpu"] is True
    assert "TORCH_HOME" in metadata["env_vars"]
    print("  OK")
    print()


def test_config_schema():
    """Test config dataclass and JSON Schema generation."""
    print("=" * 60)
    print("Test 2: Config schema")
    print("=" * 60)
    from cjm_media_plugin_demucs.plugin import DemucsPluginConfig
    from cjm_plugin_system.utils.validation import dataclass_to_jsonschema

    schema = dataclass_to_jsonschema(DemucsPluginConfig)
    print(json.dumps(schema, indent=2))
    assert "model" in schema["properties"]
    assert "device" in schema["properties"]
    assert "shifts" in schema["properties"]
    assert schema["properties"]["model"]["enum"] == [
        "htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra", "mdx_extra_q"
    ]
    print("  OK")
    print()


def test_lifecycle():
    """Test plugin initialization and cleanup."""
    print("=" * 60)
    print("Test 3: Lifecycle (initialize / cleanup)")
    print("=" * 60)
    from cjm_media_plugin_demucs.plugin import DemucsProcessingPlugin

    plugin = DemucsProcessingPlugin()

    # Initialize with defaults
    plugin.initialize()
    print(f"  Config: model={plugin.config.model}, device={plugin.config.device}")
    print(f"  Data dir: {plugin._data_dir}")
    assert plugin.config is not None
    assert plugin.storage is not None
    assert plugin.config.model == "htdemucs"

    # Initialize with custom config
    plugin.initialize({"model": "htdemucs_ft", "shifts": 2})
    assert plugin.config.model == "htdemucs_ft"
    assert plugin.config.shifts == 2
    print(f"  Re-initialized: model={plugin.config.model}, shifts={plugin.config.shifts}")

    # Cleanup
    plugin.cleanup()
    assert plugin._separator is None
    print("  Cleanup OK")
    print()


def test_is_available():
    """Test availability check."""
    print("=" * 60)
    print("Test 4: is_available")
    print("=" * 60)
    from cjm_media_plugin_demucs.plugin import DemucsProcessingPlugin

    plugin = DemucsProcessingPlugin()
    available = plugin.is_available()
    print(f"  Available: {available}")
    assert available is True, "Demucs should be available in this environment"
    print()


def test_get_info():
    """Test get_info action."""
    print("=" * 60)
    print("Test 5: get_info")
    print("=" * 60)
    from cjm_media_plugin_demucs.plugin import DemucsProcessingPlugin

    assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"

    plugin = DemucsProcessingPlugin()
    plugin.initialize()

    result = plugin.execute(action="get_info", file_path=str(TEST_FILE))
    print(f"  Result: {json.dumps(result, indent=2)}")
    assert "duration" in result
    assert result["duration"] > 0
    print()

    plugin.cleanup()


def test_separate_vocals():
    """Test vocals separation on real audio."""
    print("=" * 60)
    print("Test 6: separate_vocals")
    print("=" * 60)
    from cjm_media_plugin_demucs.plugin import DemucsProcessingPlugin

    assert TEST_FILE.exists(), f"Test file not found: {TEST_FILE}"

    plugin = DemucsProcessingPlugin()
    plugin.initialize()

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = plugin.execute(
            action="separate_vocals",
            input_path=str(TEST_FILE),
            output_dir=tmp_dir,
        )
        print(f"  Result: {json.dumps(result, indent=2)}")

        # Verify output
        output_path = Path(result["output_path"])
        assert output_path.exists(), f"Output file not found: {output_path}"
        assert output_path.stat().st_size > 0, "Output file is empty"
        print(f"  Output size: {output_path.stat().st_size:,} bytes")

        # Verify job in database
        job = plugin.storage.get_by_job_id(result["job_id"])
        assert job is not None, "Job not found in database"
        print(f"  Job stored: {job.job_id}")
        print(f"  Job action: {job.action}")

    plugin.cleanup()
    print()


def main():
    print()
    print("cjm-media-plugin-demucs Test Suite")
    print("=" * 60)
    print()

    test_import_and_metadata()
    test_config_schema()
    test_lifecycle()
    test_is_available()
    test_get_info()
    test_separate_vocals()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
