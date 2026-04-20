import os
import sys


APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from config_bootstrap import ensure_runtime_config_exists


def main() -> int:
    config_path = os.path.join(APP_DIR, "config.json")
    default_path = os.path.join(APP_DIR, "config.default.json")
    created = ensure_runtime_config_exists(config_path, default_path)
    if created:
        print(f"Bootstrapped runtime config from {os.path.basename(default_path)}")
    else:
        print("Runtime config already present or default template missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
