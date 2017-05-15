
def is_action_dependent(baseline):
    if (hasattr(baseline, "action_dependent") and baseline.action_dependent is True):
        return True
    return False

def is_shared_policy(algo):
    if (hasattr(algo, "shared_policy") and algo.shared_policy is True):
        return True
    return False
