"""
Integration Test: Demucs Processing Plugin via PluginManager

Verifies that the Demucs plugin can be loaded and executed via JobQueue
over the process boundary through the plugin system.

Run from the cjm-media-plugin-demucs conda environment:
    python tests_manual/test_demucs_plugin_system.py
"""

import asyncio
import os
import shutil
import sys
import tempfile

from cjm_plugin_system.core.manager import PluginManager
from cjm_plugin_system.core.queue import JobQueue, JobStatus
from cjm_plugin_system.core.scheduling import QueueScheduler


PLUGIN_NAME = "cjm-media-plugin-demucs"

# Reuse the fork's test file
TEST_FILE = "/mnt/SN850X_8TB_EXT4/Projects/GitHub/cj-mills/cjm-demucs-v4/test_files/segment_000.mp3"


def _reload_plugin(manager: PluginManager, config: dict = None):
    """Unload, re-discover, and reload the plugin with the given config."""
    manager.unload_all()
    manager.discover_manifests()
    plugin_meta = next(item for item in manager.discovered if item.name == PLUGIN_NAME)
    manager.load_plugin(plugin_meta, config or {})


async def test_discover_and_load():
    """Verify the plugin is discovered and loads via PluginManager."""
    print("=" * 60)
    print("TEST: Discover and Load via PluginManager")
    print("=" * 60)

    manager = PluginManager(scheduler=QueueScheduler())
    manager.discover_manifests()

    plugin_meta = next((item for item in manager.discovered if item.name == PLUGIN_NAME), None)
    if not plugin_meta:
        print(f"  Plugin {PLUGIN_NAME} not found in discovered manifests.")
        print("  Have you run 'cjm-ctl install' for this plugin?")
        return None
    print(f"  Discovered: {plugin_meta.name} v{plugin_meta.version}")

    if not manager.load_plugin(plugin_meta, {}):
        print("  Failed to load plugin.")
        return None
    print("  Loaded successfully")

    proxy = manager.plugins.get(PLUGIN_NAME)
    assert proxy is not None, "Plugin proxy not found after loading"
    print(f"  Proxy available: {PLUGIN_NAME}")

    print("  PASSED\n")
    return manager


async def test_get_info_via_queue(manager: PluginManager):
    """Verify get_info works via JobQueue over process boundary."""
    print("=" * 60)
    print("TEST: get_info via JobQueue")
    print("=" * 60)

    if not os.path.exists(TEST_FILE):
        print(f"  SKIPPED — {TEST_FILE} not found\n")
        return

    _reload_plugin(manager)

    queue = JobQueue(manager)
    await queue.start()

    job_id = await queue.submit(PLUGIN_NAME, action="get_info", file_path=TEST_FILE, priority=10)
    job = await queue.wait_for_job(job_id, timeout=30)

    assert job.status == JobStatus.completed, f"Expected completed, got {job.status}: {job.error}"
    result = job.result
    assert isinstance(result, dict)
    assert result["path"] == TEST_FILE
    assert result["duration"] > 0
    assert result["size_bytes"] > 0
    assert len(result["audio_streams"]) >= 1
    print(f"  Duration: {result['duration']:.1f}s")
    print(f"  Format: {result['format']}")

    await queue.stop()
    print("  PASSED\n")


async def test_separate_vocals_via_queue(manager: PluginManager):
    """Verify separate_vocals works via JobQueue over process boundary."""
    print("=" * 60)
    print("TEST: separate_vocals via JobQueue")
    print("=" * 60)

    if not os.path.exists(TEST_FILE):
        print(f"  SKIPPED — {TEST_FILE} not found\n")
        return

    tmp_dir = tempfile.mkdtemp(prefix="demucs_ps_test_")
    try:
        _reload_plugin(manager)

        if "--debug" in sys.argv:
            proxy = manager.plugins.get(PLUGIN_NAME)
            await asyncio.sleep(10)

        queue = JobQueue(manager)
        await queue.start()

        job_id = await queue.submit(
            PLUGIN_NAME,
            action="separate_vocals",
            input_path=TEST_FILE,
            output_dir=tmp_dir,
            priority=10
        )
        # Long timeout — separation takes ~1-2 min on GPU, longer on CPU
        job = await queue.wait_for_job(job_id, timeout=600)

        assert job.status == JobStatus.completed, f"Expected completed, got {job.status}: {job.error}"
        result = job.result
        assert "job_id" in result
        assert "output_path" in result
        assert "duration" in result
        assert "model" in result
        assert "stems_available" in result
        assert os.path.exists(result["output_path"])
        assert result["model"] == "htdemucs"
        assert "vocals" in result["stems_available"]

        file_size = os.path.getsize(result["output_path"])
        print(f"  Output: {os.path.basename(result['output_path'])}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Duration: {result['duration']:.1f}s")
        print(f"  Model: {result['model']}")
        print(f"  Stems: {result['stems_available']}")

        await queue.stop()
        print("  PASSED\n")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def test_unknown_action_via_queue(manager: PluginManager):
    """Verify unknown action fails correctly via JobQueue."""
    print("=" * 60)
    print("TEST: Unknown action via JobQueue")
    print("=" * 60)

    _reload_plugin(manager)

    queue = JobQueue(manager)
    await queue.start()

    job_id = await queue.submit(PLUGIN_NAME, action="unknown_action", priority=10)
    job = await queue.wait_for_job(job_id, timeout=30)

    assert job.status == JobStatus.failed
    print(f"  Correctly failed: {job.error}")

    await queue.stop()
    print("  PASSED\n")


async def run_integration():
    print()
    manager = await test_discover_and_load()
    if manager is None:
        print("Aborting — plugin not available.")
        sys.exit(1)

    await test_get_info_via_queue(manager)
    await test_separate_vocals_via_queue(manager)
    await test_unknown_action_via_queue(manager)

    manager.unload_all()
    print("=" * 60)
    print("ALL PLUGIN SYSTEM TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_integration())
