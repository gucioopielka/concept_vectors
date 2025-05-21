import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Suppressing the stupid warning

if any(env in os.environ for env in ['SLURM_JOB_ID', 'SLURM_CLUSTER_NAME']):
    # Monkey patch nnsight logger
    import logging
    import logging.handlers as _handlers

    # Replace RotatingFileHandler with something that never touches disk
    class _NoOpHandler(logging.Handler):
        def __init__(self, *args, **kw):
            super().__init__()
        def emit(self, record):
            pass

    _handlers.RotatingFileHandler = _NoOpHandler

import nnsight
if not nnsight.CONFIG.API.APIKEY:
    if os.environ.get("NNSIGHT_API_KEY"):
        nnsight.CONFIG.set_default_api_key(os.environ.get("NNSIGHT_API_KEY"))
    else:
        raise ValueError("NNSIGHT_API_KEY is not set. Please set it in your environment variables.")