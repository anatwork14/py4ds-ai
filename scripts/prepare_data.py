#!/usr/bin/env python3
"""Thin executable wrapper around the tested package CLI."""

from py4ds_ai.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
