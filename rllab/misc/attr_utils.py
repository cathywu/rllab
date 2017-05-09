
def is_action_dependent(baseline):
    if (hasattr(baseline, "action_dependent") and baseline.action_dependent is True):
        return True
    return False

def is_spatial_discounting(algo):
    if (hasattr(algo, "spatial_discounting") and algo.spatial_discounting is True):
        return True
    return False
