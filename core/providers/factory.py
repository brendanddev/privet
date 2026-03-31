
from core.providers.base import BaseProvider
from utils.logger import setup_logger

logger = setup_logger()


def get_provider(config: dict) -> BaseProvider:
    """
    Return the appropriate provider based on config.

    This is the single place where the provider decision is made.
    The rest of the app never needs to know which provider is active.

    Args:
        config (dict): App config dict from config.yaml

    Returns:
        BaseProvider: Configured provider instance
    """
    provider_name = config.get("provider", "ollama")
    logger.info(f"Loading provider: {provider_name}")

    if provider_name == "ollama":
        from core.providers.ollama import OllamaProvider
        return OllamaProvider(config)
    elif provider_name == "llamacpp":
        from core.providers.llamacpp import LlamaCppProvider
        return LlamaCppProvider(config)
    elif provider_name == "pleias":
        from core.providers.pleias import PleiasProvider
        return PleiasProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider_name}. Choose 'ollama', 'llamacpp', or 'pleias'.")