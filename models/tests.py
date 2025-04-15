import re

CHROMATIC = [
    'C', 'C#', 'D', 'D#', 'E', 'F',
    'F#', 'G', 'G#', 'A', 'A#', 'B'
]

NOTES = "ABCDEFGabcdefg"

TRIPLET_PREFIX = [f"3{a}{b}{c}" for a in NOTES for b in NOTES for c in NOTES]

EQUIVALENTS = {
    'B#': 'C', 'E#': 'F',
    'Cb': 'B', 'Fb': 'E',
    'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'C#': 'C#', 'D#': 'D#', 'F#': 'F#', 'G#': 'G#', 'A#': 'A#'
}

MAJOR_STEPS = [2, 2, 1, 2, 2, 2, 1]
MINOR_STEPS = [2, 1, 2, 2, 1, 2, 2]
MIXOLYDIAN_STEPS = [2, 2, 1, 2, 2, 1, 2]

DURATIONS = "23468"
DOT = "."
SUFFIXED = [f"{note}{duration}" for note in (NOTES) for duration in DURATIONS + DOT]


def eq_note(note):
    return EQUIVALENTS.get(note, note)


def normalize_note(token):

    token = token.strip()

    match = re.match(r'([_=^]*)([A-Ga-g])', token)
    if not match:
        return None

    accidental, letter = match.groups()
    letter = letter.upper()

    if accidental == '^':
        note = f"{letter}#"
    elif accidental == '_':
        note = f"{letter}b"
    else:
        note = letter

    return eq_note(note)


def build_scale(key):
    is_minor, is_mix = False, False
    if key.endswith('m'):
        is_minor = True
    if key.endswith('mix'):
        is_mix = True

    root = key[:1] if is_minor or is_mix else key
    root = eq_note(root)

    pattern = MAJOR_STEPS
    if is_mix:
        pattern = MIXOLYDIAN_STEPS
    elif is_minor:
        pattern = MINOR_STEPS

    try:
        idx = CHROMATIC.index(root)
    except ValueError:
        # print(f"Unknown root: {root}")
        return []

    scale = [CHROMATIC[idx]]
    for step in pattern:
        idx = (idx + step) % len(CHROMATIC)
        scale.append(CHROMATIC[idx])
    return [eq_note(n) for n in scale]


def evaluate_key(sample, key):
    scale = build_scale(key)
    if not scale:
        return 0.0

    in_key = {}
    out_key = {}
    total = 0

    for raw in sample.split():
        note = normalize_note(raw)
        if note is None:
            continue
        total += 1

        if note in scale:
            in_key[note] = in_key.get(note, 0) + 1
        else:
            out_key[note] = out_key.get(note, 0) + 1


    # output frequencies of in/out key notes

    # print(f"\nKEY: {key}\nSCALE: {scale}")
    # if in_key:
    #     # print("\nValid note frequencies:")
    #     for note, count in sorted(in_key.items()):
    #         print(f"{note}: {count}")

    # if out_key:
    #     # print("\nInvalid note frequencies:")
    #     for note, count in sorted(out_key.items()):
    #         print(f"{note}: {count}")

    score = sum(in_key.values()) / total if total > 0 else 0.0

    return score


def evaluate_time_signature(sample, length, time_sig):
    # print(f"Evaluating time signature: {time_sig}, length: {length}")
    
    # Parse base note length (e.g., "1/8" -> 0.125)
    length = eval(length)  # Convert string fraction to float
    
    # Convert time signature to total beats per measure
    if time_sig == "C":
        beats_per_bar = 4/4  # 4/4
    elif time_sig == "C|":
        beats_per_bar = 2/2  # 2/2
    else:
        beats_per_bar = eval(time_sig)  # e.g., "3/4" -> 0.75
    
    bars = [bar.strip() for bar in sample.split('|') if bar.strip()]
    correct = 0
    
    for i, bar in enumerate(bars):
        # print(f"\n{i} Bar: {bar}")
        total_duration = 0.0

        #initial two are definiting time signature and length
        i = 2
        
        while i < len(bar):
            # Handle triplets (e.g., "3DEF")
            if i + 3 <= len(bar) and bar[i] == '3' and bar[i+1] in NOTES and bar[i+2] in NOTES:
                total_duration += 2 * length  # 3 notes in time of 2
                i += 3
            
            # Handle dotted notes (e.g., ".D")
            elif i + 1 < len(bar) and bar[i+1] in NOTES and bar[i] == DOT:
                total_duration += 1.5 * length  # Original + half
                i += 2
            
            # Handle regular notes with duration (e.g., "C2")
            elif i + 1 < len(bar) and bar[i] in NOTES and bar[i+1] in DURATIONS:
                dur = int(bar[i+1])
                total_duration += length * (dur)  # e.g., "C2" in 1/8 base = 4/2=2 units
                i += 2
            
            # Handle simple notes (e.g., "C")
            elif bar[i] in NOTES:
                total_duration += length
                i += 1
            
            # Skip rests/other characters
            else:
                i += 1
        
        # Compare with tolerance for floating-point precision
        if abs(total_duration - beats_per_bar) < 0.001:
            correct += 1
            # print(f"✓ Correct duration: {total_duration}")
        # else:
        #     print(f"✗ Incorrect duration: {total_duration} (expected {beats_per_bar})")
    
    return correct / len(bars) if bars else 0.0