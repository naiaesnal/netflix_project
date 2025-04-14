import pandas as pd
import cornac


def load_netflix_data(variant='small', fmt='UIR'):
    """Load Netlflix dataset from Cornac."""
    feedback = cornac.datasets.netflix.load_feedback(fmt=fmt, variant=variant)
    print('Loaded dataset:')
    print('Type:', type(feedback))
    print('Number of entries:', len(feedback))
    return feedback
