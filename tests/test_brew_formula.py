"""scripts/brew_formula.py: file choice per package and the rendered formula.
No network: the PyPI and uv steps are not exercised here."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "brew_formula.py"


def _load():
    spec = importlib.util.spec_from_file_location("_brew_formula", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf = _load()


def _files(*names):
    return [{"filename": n, "url": f"https://files/{n}",
             "packagetype": "sdist" if n.endswith(".tar.gz") else "bdist_wheel",
             "digests": {"sha256": n}} for n in names]


def _pick(*names):
    chosen = bf.pick_file(_files(*names), "3.13")
    return chosen and chosen["filename"]


def test_native_wheel_prefers_the_newest_macos_tag_up_to_the_floor():
    assert _pick("mlx-1-cp313-cp313-macosx_14_0_arm64.whl",
                 "mlx-1-cp313-cp313-macosx_26_0_arm64.whl",
                 "mlx-1-cp313-cp313-macosx_27_0_arm64.whl",
                 "mlx-1.tar.gz") == "mlx-1-cp313-cp313-macosx_26_0_arm64.whl"


def test_native_over_abi3_over_pure():
    assert _pick("a-1-py3-none-any.whl",
                 "a-1-cp39-abi3-macosx_11_0_arm64.whl",
                 "a-1-cp313-cp313-macosx_11_0_arm64.whl") \
        == "a-1-cp313-cp313-macosx_11_0_arm64.whl"
    assert _pick("a-1-py3-none-any.whl",
                 "a-1-cp39-abi3-macosx_11_0_universal2.whl") \
        == "a-1-cp39-abi3-macosx_11_0_universal2.whl"


def test_unusable_wheels_are_skipped():
    assert _pick("a-1-cp312-cp312-macosx_14_0_arm64.whl",
                 "a-1-cp313-cp313-macosx_14_0_x86_64.whl",
                 "a-1-cp313-cp313-manylinux_2_28_aarch64.whl",
                 "a-1-cp313-cp313t-macosx_14_0_arm64.whl",
                 "a-1-cp314-abi3-macosx_14_0_arm64.whl",
                 "a-1.tar.gz") == "a-1.tar.gz"


def test_compressed_platform_tags():
    assert _pick("a-1-py2.py3-none-any.whl") == "a-1-py2.py3-none-any.whl"
    assert _pick("a-1-cp313-cp313-macosx_10_9_x86_64.macosx_11_0_arm64.whl") \
        == "a-1-cp313-cp313-macosx_10_9_x86_64.macosx_11_0_arm64.whl"


def test_no_wheel_and_no_sdist():
    assert _pick("a-1-cp313-cp313-win_amd64.whl") is None


def test_parse_pins_reads_compiled_requirements():
    text = "# header\nmlx==0.32.1\n    # via gmlx\nruamel-yaml==0.18.6\n\n"
    assert bf.parse_pins(text) == [("mlx", "0.32.1"), ("ruamel-yaml", "0.18.6")]


def test_render_lists_every_resource_once():
    source = {"name": "gmlx", "url": "https://files/gmlx-1.tar.gz",
              "sha256": "0" * 64}
    resources = [{"name": n, "url": f"https://files/{n}.whl", "sha256": "1" * 64}
                 for n in ("numpy", "mlx", "addict")]
    text = bf.render(source, resources, "3.13")
    assert text.count('depends_on "python@3.13"') == 1
    assert 'python3.13", system_site_packages: false' in text
    blocks = [line for line in text.splitlines() if line.startswith("  resource ")]
    assert blocks == ['  resource "addict" do', '  resource "mlx" do',
                      '  resource "numpy" do']
    assert '"#{buildpath}[all]"' in text
    assert "depends_on macos: :tahoe" in text
