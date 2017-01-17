import os
import sys
import json
import numpy


conf_file = sys.argv[1] if len(sys.argv) > 1 else None
with open(conf_file) as f:
    conf = json.load(f)

num_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 25
dir_json = sys.argv[3] if len(sys.argv) > 3 else 'json/'
rng = numpy.random.RandomState(conf['rng_seed'])

model_name = conf['model_name']
if not os.path.exists(dir_json):
    os.makedirs(dir_json)


def random_init_string(sparse=False):
    if rng.randint(2) and sparse:
        sparse_init = rng.randint(10, 30)
        return "sparse_init: " + str(sparse_init)
    irange = 10. ** rng.uniform(-2.3, -1.)
    return "irange: " + str(irange)


def opt_param(max_val=1.0, min_val=0.):
    if rng.randint(2):
        return 0
    return rng.uniform(min_val, max_val)

for job_id in range(num_jobs):
    conf['model_name'] = '%s_%02i' % (model_name, job_id)
    if os.path.isfile(conf['model_name']):
        print('%s already exists, skipping...' % (conf['model_name']))
        continue

    conf['init_ranges'] = [10. ** rng.uniform(-3, -1.) for _ in range(3)]
    # Regularization params
    # http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    if rng.randint(2):
        conf['max_norms'] = [rng.uniform(5., 20.) for _ in range(3)]
    conf['hidden_size'] = [64, 128, 256, 512][rng.randint(4)]
    conf['dropout'] = rng.uniform(0.3, 0.7)
    conf['learning_rate'] = 10. ** rng.uniform(-3., -1.)

    with open(os.path.join(dir_json, conf['model_name'] + '.json'), 'w') as f:
        json.dump(conf, f, indent=4)
