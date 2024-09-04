import kopf
from typing import AsyncIterator
import logging

_LOGGER = logging.getLogger(__name__)


class WebhookServiceServer(kopf.WebhookServer):

    def __init__(self, *, namespace, name, **kwargs):
        kwargs["extra_sans"] = [f"{name}.{namespace}.svc"]
        super().__init__(**kwargs)
        self.namespace = namespace
        self.name = name

    async def __call__(self, fn: kopf.WebhookFn) -> AsyncIterator[kopf.WebhookClientConfig]:
        async for client_config in super().__call__(fn):
            client_config["service"] = kopf.WebhookClientConfigService(
                namespace=self.namespace,
                name=self.name,
                path="",
            )
            client_config["url"] = None
            # _LOGGER.debug(f"client_config: {client_config}")
            yield client_config
