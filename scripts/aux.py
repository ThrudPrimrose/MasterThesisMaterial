def darken_hex_color(hex_color, factor=0.7):
    """
    Darken a hex color by a given factor.
    
    Parameters:
    - hex_color: The original color in HEX format.
    - factor: A value between 0 and 1. A smaller value will result in a darker color.
    
    Returns:
    - The darkened color in HEX format.
    """
    # Convert HEX to RGB
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    # Darken the RGB components
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    # Convert RGB back to HEX
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
