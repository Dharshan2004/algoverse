"""Read preview CSVs without coercing source text or guessing label truthiness."""
import pandas as pd

REQUIRED = {'user_prompt','model_response','is_jailbroken'}

def read_preview(path):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f'Missing preview columns: {sorted(missing)}')
    for field in ['user_prompt','model_response']:
        if not frame[field].map(lambda value: bool(value.strip())).all():
            raise ValueError(f'Empty {field} in preview')
    if not frame['is_jailbroken'].isin(['True','False']).all():
        raise ValueError('Preview labels must be exactly True or False')
    frame['is_jailbroken'] = frame['is_jailbroken'].map({'True':True,'False':False})
    return frame
