import os
import shutil


def ensure_runtime_config_exists(config_path: str, default_config_path: str) -> bool:
    target = os.path.abspath(str(config_path or "").strip())
    default = os.path.abspath(str(default_config_path or "").strip())
    if not target or not default:
        return False
    if os.path.exists(target):
        return False
    if not os.path.exists(default):
        return False
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    shutil.copy2(default, target)
    return True
