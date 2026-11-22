import os

_env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_file):
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = os.path.expandvars(os.path.expanduser(value.strip().strip("'\"")))
            if value and not os.environ.get(key):
                os.environ[key] = value

os.environ.setdefault("HF_HOME", os.path.expanduser("~/orcd/pool/.cache/huggingface"))
