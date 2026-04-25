import joblib
from pprint import pprint

p = joblib.load('dropdown_options.pkl')
print('keys:', list(p.keys()))
print('job_title options:')
pprint(p.get('job_title', []))
