import datetime as dt

def loggg(f):
    def wrapper(df, *args, **kwargs):
        tic = dt.datetime.now()
        result = f(df, *args, **kwargs)
        toc = dt.datetime.now()
        print(f"{f.__name__} \n took={toc-tic} shape={result.shape}")
        return result
    return wrapper

@loggg
def start_pipeline(df):
    return df.copy()