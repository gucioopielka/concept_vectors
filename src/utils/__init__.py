import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Suppressing the stupid warning

import nnsight
if not nnsight.CONFIG.API.APIKEY:
    nnsight.CONFIG.set_default_api_key(os.environ["NNSIGHT_API_KEY"])