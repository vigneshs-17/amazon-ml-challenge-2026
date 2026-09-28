"""Comprehensive Secret and Privacy Audit Tool."""

import os
import re

PATTERNS = [
    r"AKIA[0-9A-Z]{16}",  # AWS access key ID
    r"ghp_[a-zA-Z0-9]{36}",  # GitHub personal access token
    r"github_pat_[a-zA-Z0-9_]{82}",  # GitHub fine-grained token
    r"sk-[a-zA-Z0-9]{32,}",  # OpenAI / generic secret key
    r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",  # Private keys
    r'(?:password|passwd|pwd)\s*[:=]\s*["\'][^"\']+["\']',
    r'(?:api_key|apikey|secret_key|access_token|bearer\s+[a-zA-Z0-9_\-\.]+)\s*[:=]\s*["\'][^"\']+["\']',
    r"[cC]:\\Users\\[a-zA-Z0-9_\-\.]+",  # C:\Users\... personal paths
    r"[dD]:\\Users\\[a-zA-Z0-9_\-\.]+",  # D:\Users\... personal paths
    r"[dD]:\\AmazonMLChallenge2026",  # D:\AmazonMLChallenge2026 hardcoded paths
]

REGEX = re.compile("|".join(PATTERNS), re.IGNORECASE)

# Explicit exclusion for files not intended for Git
EXCLUDE_DIRS = {".git", "__pycache__", "output", "data", "scratch", "logs", ".venv", "venv"}


def audit():
    flagged = []
    scanned = 0
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if f in ["check_repo_secrets.py"]:
                continue
            if f.endswith((".py", ".md", ".json", ".txt", ".sh", ".bat", ".yaml", ".yml", ".csv", ".toml")):
                p = os.path.normpath(os.path.join(root, f))
                scanned += 1
                with open(p, "r", encoding="utf-8", errors="ignore") as fp:
                    for line_no, line in enumerate(fp, 1):
                        m = REGEX.search(line)
                        if m:
                            flagged.append((p, line_no, m.group(0), line.strip()[:100]))

    print(f"Scanned {scanned} files intended for Git.")
    print(f"Total potential secret / absolute personal path lines flagged: {len(flagged)}")
    for p, line_num, match, snippet in flagged:
        print(f"  {p}:{line_num} [{match}] -> {snippet}")
    return len(flagged)


if __name__ == "__main__":
    audit()
