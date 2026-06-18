"""
Isochronic timing engine — ensures dubbed Hindi audio fits original segment duration.
Strategy: estimate TTS output length from character count, compute speed factor.
"""

HINDI_CHARS_PER_SECOND = 8.0   # average Hindi TTS speaking rate
MIN_SPEED = 0.75
MAX_SPEED = 1.6


def estimate_hindi_duration(hindi_text):
    chars = len(hindi_text.replace(' ', '').replace('।', ''))
    return max(0.3, chars / HINDI_CHARS_PER_SECOND)


def compute_speed_factor(original_duration, hindi_text):
    """
    Returns how fast TTS should speak to match the original clip duration.
    Clamped to [0.75, 1.6] — outside that range speech quality degrades.
    """
    estimated = estimate_hindi_duration(hindi_text)
    if original_duration <= 0:
        return 1.0
    factor = estimated / original_duration
    return max(MIN_SPEED, min(MAX_SPEED, factor))
