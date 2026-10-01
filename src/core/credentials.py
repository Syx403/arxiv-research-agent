"""Access only ARA's named API-key item in the macOS login Keychain.

Secrets never travel through shell arguments or command output. Other platforms
continue to use environment variables / the existing local settings file.
"""
from __future__ import annotations

import ctypes
import sys

SERVICE = b"arxiv-research-agent"
ACCOUNT = b"DEEPSEEK_API_KEY"
_NOT_FOUND = -25300
_DUPLICATE = -25299


def _frameworks():
    security = ctypes.CDLL("/System/Library/Frameworks/Security.framework/Security")
    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    security.SecKeychainFindGenericPassword.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecKeychainAddGenericPassword.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32,
        ctypes.c_char_p, ctypes.c_uint32, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    security.SecKeychainItemModifyAttributesAndData.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
    ]
    security.SecKeychainItemFreeContent.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    core.CFRelease.argtypes = [ctypes.c_void_p]
    return security, core


def read_deepseek_key() -> str | None:
    if sys.platform != "darwin":
        return None
    security, core = _frameworks()
    length, data, item = ctypes.c_uint32(), ctypes.c_void_p(), ctypes.c_void_p()
    status = security.SecKeychainFindGenericPassword(
        None, len(SERVICE), SERVICE, len(ACCOUNT), ACCOUNT,
        ctypes.byref(length), ctypes.byref(data), ctypes.byref(item),
    )
    if status == _NOT_FOUND:
        return None
    if status != 0:
        raise RuntimeError(f"ARA Keychain read failed (OSStatus {status}); no secret was logged.")
    try:
        return ctypes.string_at(data, length.value).decode("utf-8")
    finally:
        security.SecKeychainItemFreeContent(None, data)
        if item:
            core.CFRelease(item)


def save_deepseek_key(secret: str) -> None:
    if sys.platform != "darwin":
        raise RuntimeError("Native Keychain setup requires macOS.")
    if not secret.strip() or any(c.isspace() for c in secret):
        raise ValueError("The API key must be nonempty and contain no whitespace.")
    security, core = _frameworks()
    encoded = secret.encode("utf-8")
    item = ctypes.c_void_p()
    status = security.SecKeychainAddGenericPassword(
        None, len(SERVICE), SERVICE, len(ACCOUNT), ACCOUNT, len(encoded), encoded,
        ctypes.byref(item),
    )
    if status == _DUPLICATE:
        # Find the item reference only; do not read its previous secret.
        status = security.SecKeychainFindGenericPassword(
            None, len(SERVICE), SERVICE, len(ACCOUNT), ACCOUNT, None, None,
            ctypes.byref(item),
        )
        if status == 0:
            status = security.SecKeychainItemModifyAttributesAndData(
                item, None, len(encoded), encoded,
            )
    if item:
        core.CFRelease(item)
    if status != 0:
        raise RuntimeError(f"ARA Keychain save failed (OSStatus {status}); no secret was logged.")
