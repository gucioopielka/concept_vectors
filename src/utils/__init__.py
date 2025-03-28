import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Suppressing the stupid warning

import nnsight
if not nnsight.CONFIG.API.APIKEY:
    if os.environ.get("NNSIGHT_API_KEY"):
        nnsight.CONFIG.set_default_api_key(os.environ.get("NNSIGHT_API_KEY"))
    else:
        raise ValueError("NNSIGHT_API_KEY is not set. Please set it in your environment variables.")