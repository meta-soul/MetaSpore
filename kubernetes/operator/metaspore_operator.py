import os
import kopf
import asyncio
from kubernetes_asyncio import client, config

@kopf.on.login()
def login_fn(**kwargs):
    print(f'Logging in ... {kwargs}')
    return kopf.login_via_client(**kwargs)

@kopf.on.startup()
async def startup_fn(logger, **kwargs):
    logger.info(f'Starting ... {kwargs}')
    if 'INCLUSTER' in os.environ:
        config.load_incluster_config()
    else:
        await config.load_kube_config()
    await asyncio.sleep(1)

@kopf.on.cleanup()
async def cleanup_fn(logger, **kwargs):
    logger.info('Cleaning up in 3s...')
    await asyncio.sleep(3)

@kopf.on.create('sourcetables')
def create_fn(spec, name, namespace, logger, **kwargs):
    logger.info(f'{spec}, {name}, {namespace}')