#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from arena.experimental_models import refresh_models, list_experimental_models
refresh_models()
models = list_experimental_models()
print(f'Total models: {len(models)}')
for m in sorted(models):
    print(f'  - {m}')
