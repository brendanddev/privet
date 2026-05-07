from privet.providers.base import BaseProvider

def get_provider(config: dict) -> BaseProvider:
    provider_name = config.get("provider", "ollama")

    if provider_name == "ollama":
        pass

    if provider_name == "llamacpp":
        pass

    raise ValueError(f"Unknown provider: {provider_name}")