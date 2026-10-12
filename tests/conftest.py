"""Suite-wide defaults.

The upstream tests exercise the per-client temp workspace. Persistent conversations (which move
agy's HOME into plugin data) are opt-out at runtime but opted out here by default, so those tests
keep testing what they were written for; tests/test_persistent_conversations.py turns it on.
"""
import os

os.environ.setdefault("ANTIGRAVITY_PERSIST_CONVERSATIONS", "0")
