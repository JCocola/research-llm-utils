from pathlib import Path


def load_secrets(secrets_file: str) -> dict[str, str]:
    """
    Load API keys from a secrets file containing KEY=VALUE pairs.

    Args:
        secrets_file (str): Path to the secrets file.

    Returns:
        dict[str, str]: Dictionary mapping secret names to their values.
    """
    secrets_path = Path(secrets_file).resolve()
    secrets = {}

    if secrets_path.exists():
        with secrets_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                secrets[key.strip()] = value.strip()
    else:
        print(f"Secrets file '{secrets_file}' does not exist.")

    return secrets


# Example usage:
# secrets = load_secrets("/path/to/SECRETS")
# os.environ["OPENAI_API_KEY"] = secrets.get("OPENAI_API_KEY1", "")
