#!/usr/bin/env python
"""Check consistent coding style."""
import os
import sys
import logging
import subprocess


def get_commit_files():
    """Get the committed files and return them as a list."""
    cmd = ["git", "diff", "--cached", "--name-only"]
    commit_files = subprocess.check_output(cmd, encoding="utf-8")
    return commit_files.splitlines()


def _check_py_style(filename):
    """Check python coding style using pep8"""
    ret = subprocess.call(["pep8", filename])
    if ret != 0:
        logging.info("Not comply to pep8: %s", filename)
        return False

    ret = subprocess.call(["pylint", "-E", filename])
    if ret != 0:
        logging.warning("Not comply to pylint -E: %s", filename)
        return False

    logging.info("PASS pycheck: %s.", filename)
    return True


def _is_style_consistent(filename):
    """Return True if coding style is met and False otherwise"""
    _path, ext = os.path.splitext(filename)
    if ext == ".py":
        if os.path.isfile(filename):
            return _check_py_style(filename)
        # File is deleted, so return True unconditionally.
        logging.warning("Do not check deleted file: %s", filename)
        return True
    logging.info("No style checker for %s", filename)
    return True


def main():
    """Executable entry"""
    os.environ["PYTHONPATH"] = "."
    commit_files = get_commit_files()
    fail_count = 0
    for filename in commit_files:
        if not _is_style_consistent(filename):
            fail_count += 1
    return fail_count


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(message)s",
                        level=logging.INFO)
    sys.exit(main())
