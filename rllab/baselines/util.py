
def is_action_dependent(baseline):
    if (hasattr(baseline, "action_dependent") and baseline.action_dependent is True):
        return True
    return False
