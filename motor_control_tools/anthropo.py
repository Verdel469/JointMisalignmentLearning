def arm_length(height):
    """Compute arm length based on height. 
    References : Anthropometric table of Winter

    Args:
        height (float): Height of the subject in meters

    Returns:
        float: Estimated length of the arm in meters
    """
    return 0.186*height

def forearm_length(height):
    """Compute forearm length based on height. 
    References : Anthropometric table of Winter

    Args:
        height (float): Height of the subject in meters

    Returns:
        float: Estimated length of the forearm in meters
    """
    return 0.146*height

def hand_length(height):
    """Compute hand length based on height. 
    References : Anthropometric table of Winter

    Args:
        height (float): Height of the subject in meters

    Returns:
        float: Estimated length of the hand in meters
    """
    return 0.108*height