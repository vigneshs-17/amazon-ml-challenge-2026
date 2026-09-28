"""Repository scan and classification script."""

import os

scan_results = {"keep_git": [], "keep_local_gitignore": [], "safe_delete": [], "manual_review": []}

total_files = 0
total_dirs = 0

for root, dirs, files in os.walk("."):
    total_dirs += len(dirs)
    for f in files:
        total_files += 1
        p = os.path.normpath(os.path.join(root, f))

        # Categorize
        if any(part in p for part in ["__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"]) or f.endswith(
            (".pyc", ".pyo", ".tmp", ".temp", ".rev.tmp", "Thumbs.db", ".DS_Store")
        ):
            scan_results["safe_delete"].append(p)
        elif any(
            p.startswith(prefix)
            for prefix in [
                os.path.normpath("data/raw"),
                os.path.normpath("data/interim"),
                os.path.normpath("output/chunks"),
                os.path.normpath("output/deadline_fallback"),
                os.path.normpath("output/deadline_final"),
                os.path.normpath("output/final"),
                os.path.normpath("logs"),
                os.path.normpath("scratch"),
                os.path.normpath("output/candidate_pairs.tsv"),
                os.path.normpath("output/matching_results.tsv"),
            ]
        ) or f.endswith((".joblib", ".parquet", ".pkl")):
            scan_results["keep_local_gitignore"].append(p)
        elif any(
            p.startswith(prefix)
            for prefix in [
                os.path.normpath("src"),
                os.path.normpath("tests"),
                os.path.normpath("experiments"),
                os.path.normpath("reports"),
                os.path.normpath(".github"),
            ]
        ) or f in [
            "README.md",
            "Documentation_template.md",
            "requirements.txt",
            "requirements-dev.txt",
            ".gitignore",
            "LICENSE",
        ]:
            scan_results["keep_git"].append(p)
        else:
            scan_results["manual_review"].append(p)

print(f"Total scanned: {total_files} files across {total_dirs} directories")
print(f"Safe to delete: {len(scan_results['safe_delete'])}")
print(f"Keep locally + gitignore: {len(scan_results['keep_local_gitignore'])}")
print(f"Keep in git: {len(scan_results['keep_git'])}")
print(f"Manual review: {len(scan_results['manual_review'])}")

if scan_results["manual_review"]:
    print("\nManual review items:")
    for item in scan_results["manual_review"][:50]:
        print(" ", item)
