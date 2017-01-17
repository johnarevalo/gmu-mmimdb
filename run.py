import json
import os
import sys
import train

conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    params = json.load(f)

trainer = getattr(train, params['model_class'])(params)
trainer.train()
trainer.evaluate()
