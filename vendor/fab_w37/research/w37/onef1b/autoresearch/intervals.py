"""Keep absolute nanosecond timestamps integral even at group boundaries."""

def preceding_end(frame,keys,column='completion_ns'):
    return frame[column].astype('Int64').groupby([frame[k] for k in keys],sort=False).shift()
