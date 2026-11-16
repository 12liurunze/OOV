"""Install two frozen SGLang source trees from the verified public base and overlays."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "runtime"
DEFAULT_OUTPUT = ROOT / "generated" / "runtime"
BUNDLED_ARCHIVE = SOURCES / "vendor" / "original_runtime.tar.gz"
ARMS = ("native", "repair")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_manifest():
    record = json.loads((SOURCES / "manifest.json").read_text(encoding="utf-8"))
    if record["schema_version"] != 1 or set(record["arms"]) != set(ARMS):
        raise ValueError("Unsupported SGLang runtime manifest")
    return record


def safe_path(name):
    path = PurePosixPath(name)
    if (
        not path.parts
        or path.parts[0] != "sglang"
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
    ):
        raise ValueError(f"Unsafe SGLang source path {name!r}")
    return path


def file_inventory(parent):
    package = parent / "sglang"
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"No regular SGLang package directory at {package}")
    result = {}
    for path in package.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Runtime must use regular files and directories: {path}")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo"):
            result[path.relative_to(parent).as_posix()] = file_sha256(path)
    return result


def identity(inventory):
    return {"count": len(inventory), "sha256": canonical_sha256(inventory)}


def verify_arm(path, expected):
    sources = file_inventory(path)
    python = {name: digest for name, digest in sources.items() if name.endswith(".py")}
    actual = {"all": identity(sources), "python": identity(python)}
    if actual != expected["inventory"]:
        raise ValueError(f"Installed source inventory mismatch at {path}: {actual}")
    return actual


def check_archive(archive, expected):
    if archive.stat().st_size != expected["bytes"] or file_sha256(archive) != expected["sha256"]:
        raise ValueError(f"Unexpected base archive size or SHA-256: {archive}")


def download(expected, destination):
    remaining = expected["bytes"]
    with urlopen(expected["url"], timeout=60) as response, destination.open("xb") as output:
        while block := response.read(1024 * 1024):
            remaining -= len(block)
            if remaining < 0:
                raise ValueError("Base download exceeds the frozen archive size")
            output.write(block)
    check_archive(destination, expected)


def unpack(archive, parent):
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = safe_path(member.name)
            relative = path.as_posix()
            if relative in seen or not (member.isfile() or member.isdir()):
                raise ValueError(f"Invalid or duplicate base archive member: {member.name}")
            seen.add(relative)
            target = parent.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(member) as stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                target.chmod(member.mode & 0o777)


def apply_overlay(parent, overlay, recipe):
    for name in recipe["remove"]:
        target = parent.joinpath(*safe_path(name).parts)
        target.unlink()
    expected = recipe["files"]
    supplied = file_inventory(overlay)
    if supplied != expected:
        raise ValueError(f"Overlay membership or SHA-256 mismatch: {overlay}")
    for name in expected:
        path = safe_path(name)
        target = parent.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(overlay.joinpath(*path.parts), target)


def verify_installed(output):
    """Verify the exact source trees before importing either runtime."""
    record = read_manifest()
    return {
        "schema_version": 1,
        "base_archive_sha256": record["base"]["sha256"],
        "runtime_manifest_sha256": file_sha256(SOURCES / "manifest.json"),
        "arms": {
            arm: verify_arm(output / arm, record["arms"][arm])
            for arm in ARMS
        },
    }


def install(output, local_archive=None):
    record = read_manifest()
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Destination already exists: {output}. Use --verify-installed to check it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sglang-oov-", dir=output.parent) as temporary:
        workspace = Path(temporary)
        archive = local_archive
        if archive is None and BUNDLED_ARCHIVE.is_file():
            archive = BUNDLED_ARCHIVE
        if archive is None:
            archive = workspace / "base.tar.gz"
            download(record["base"], archive)
        else:
            check_archive(archive, record["base"])
        staging = workspace / "runtime"
        native = staging / "native"
        unpack(archive, native)
        if identity(file_inventory(native)) != record["base"]["inventory"]:
            raise ValueError("Extracted base source inventory differs from the manifest")
        apply_overlay(native, SOURCES / "native", record["arms"]["native"])
        repair = staging / "repair"
        shutil.copytree(native, repair)
        apply_overlay(repair, SOURCES / "repair", record["arms"]["repair"])
        receipt = verify_installed(staging)
        (staging / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"Runtime destination appeared during install: {output}")
        staging.rename(output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Install from the frozen archive already present on disk")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-installed", action="store_true", help="Check both source inventories without writing")
    args = parser.parse_args(argv)
    if args.verify_installed and args.archive:
        parser.error("--archive applies only to installation")
    try:
        receipt = verify_installed(args.output) if args.verify_installed else install(args.output, args.archive)
    except (ValueError, OSError, tarfile.TarError) as exc:
        parser.exit(1, f"Runtime verification failed: {exc}\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
