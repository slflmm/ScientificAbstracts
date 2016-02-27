TARGET_SUFFIXES_1a = [
    ['sses', 'ss'],
    ['ies', 'i'],
    ['ss', 'ss'],
    ['s', '']
]

TARGET_SUFFIXES_2 = [
    ['ational', 'ate'],
    ['tional', 'tion'],
    ['enci', 'ence'],
    ['anci', 'ance'],
    ['izer', 'ize'],
    ['abli', 'able'],
    ['alli', 'al'],
    ['entli', 'ent'],
    ['eli', 'e'],
    ['ousli', 'ous'],
    ['ization', 'ize'],
    ['ation', 'ate'],
    ['ator', 'ate'],
    ['alism', 'al'],
    ['iveness', 'ive'],
    ['fulness', 'ful'],
    ['ousness', 'ous'],
    ['aliti', 'al'],
    ['iviti', 'ive'],
    ['biliti', 'ble']
]

TARGET_SUFFIXES_3 = [
    ['icate', 'ic'],
    ['ative', ''],
    ['alize', 'al'],
    ['iciti', 'ic'],
    ['ical', 'ic'],
    ['ful', ''],
    ['ness', '']
]

TARGET_SUFFIXES_4 = [
    ['al', ''],
    ['ance', ''],
    ['ence', ''],
    ['er', ''],
    ['ic', ''],
    ['able', ''],
    ['ible', ''],
    ['ant', ''],
    ['ement', ''],
    ['ment', ''],
    ['ent', ''],
    ['tion', 't'],
    ['sion', 's'],
    ['ou', ''],
    ['ism', ''],
    ['ate', ''],
    ['iti', ''],
    ['ous', ''],
    ['ive', ''],
    ['ize', '']
]


def patternize(word, sequenced):
    '''
    Translate word to its [C]VCVC...[V] pattern.
    C = consonant sequence
    V = vowel sequence
    The letter 'y' is treated as a vowel if preceded by
    a consonant, otherwise 'y' is a consonant.
    '''
    pattern = []
    for char in word:
        letter_type = ''
        if char in 'aeiou':
            letter_type = 'v'
        elif char == 'y':
            letter_type = 'c' if (pattern and pattern[-1] == 'v') else 'v'
        else:
            letter_type = 'c'
        if not pattern or not sequenced or letter_type != pattern[-1]:
            pattern.append(letter_type)

    return ''.join(pattern)


def find_measure(word):
    '''
    Given a word, returns its measure.
    A word can be written as [C]VCVC...[V] = [C](VC)^m[V]
    Each (VC) is a consonant sequence (may contain >1 'C')
    Measure = m
    '''
    pattern = patternize(word, True)
    return pattern.count('vc')


def stem(word, suffix):
    '''
    Returns stem of word given suffix
    '''
    return word[:-len(suffix)]


def rsuffix(word, suffix, replacement):
    '''
    Helper method for suffix replacement.
    '''
    return stem(word, suffix) + replacement


def vowel_in_stem(word, suffix):
    '''
    Helper method; returns true if there is a vowel in the word's stem
    '''
    return 'v' in patternize(stem(word, suffix), False)


def double_end(word):
    '''
    Helper method; returns true if word ends with double letter
    '''
    return len(word) >= 2 and word[-1] == word[-2]


def cvc(word, suffix):
    '''
    Returns true if the stem ends with cvc, where second c is not 'w', 'x', or 'y'
    '''
    st = stem(word, suffix)
    wxy_ending = st.endswith('w') or st.endswith('x') or st.endswith('y')
    return patternize(st, False).endswith('cvc') and not wxy_ending


def simple_loop(word, suffixes, m_greater_than):
    '''
    Given a word and list of suffixes and their replacements, replaces the suffix
    at the end of the word if the stem has measure greater than m
    '''
    for suffix, target in suffixes:
        if word.endswith(suffix) and find_measure(stem(word, suffix)) > m_greater_than:
            word = rsuffix(word, suffix, target)
            break
    return word


def step_one(word):
    '''
    Step 1 deals with plurals and past participles.
    Also turns terminal 'y' to 'i' when there is another vowel in the stem.
    It is also the most complicated.
    '''
    # step 1a
    word = simple_loop(word, TARGET_SUFFIXES_1a, -1)

    # step 1b
    successful = False
    if word.endswith('eed'):
        if find_measure(stem(word, 'eed')) > 0:
            word = rsuffix(word, 'eed', 'ee')
    elif word.endswith('ed') and vowel_in_stem(word, 'ed'):
        word = rsuffix(word, 'ed', '')
        successful = True
    elif word.endswith('ing') and vowel_in_stem(word, 'ing'):
        word = rsuffix(word, 'ing', '')
        successful = True

    # step 1b.2
    if successful:
        if word.endswith('at'):
            word = rsuffix(word, 'at', 'ate')
        elif word.endswith('bl'):
            word = rsuffix(word, 'bl', 'ble')
        elif word.endswith('iz'):
            word = rsuffix(word, 'iz', 'ize')
        elif double_end(word) and not (word[-1] in ['l', 's', 'z']):
            word = word[:-1]
        elif find_measure(word) == 1 and cvc(word, ''):
            word = word + 'e'

    # step 1c
    if word.endswith('y') and vowel_in_stem(word, 'y'):
        word = rsuffix(word, 'y', 'i')

    return word


def step_two(word):
    '''
    Step 2 maps double suffixes to single ones
    '''
    return simple_loop(word, TARGET_SUFFIXES_2, 0)


def step_three(word):
    '''
    Step 3 deals with common suffixes
    '''
    return simple_loop(word, TARGET_SUFFIXES_3, 0)


def step_four(word):
    '''
    Step 4 takes off the remaining extras
    '''
    return simple_loop(word, TARGET_SUFFIXES_4, 1)


def step_five(word):
    '''
    Step 5 removes an extra 'e' if need be
    '''
    # 5a
    if word.endswith('e') and find_measure(stem(word, 'e')) > 1:
        word = rsuffix(word, 'e', '')
    elif word.endswith('e') and find_measure(stem(word, 'e')) == 1 and not cvc(word, 'e'):
        word = rsuffix(word, 'e', '')

    # 5b
    if word.endswith('l') and find_measure(stem(word, 'l')) > 1 and double_end(word):
        word = word[:-1]

    return word


def porter_stem(word):
    '''
    Returns the stem of 'word' according to Porter's algorithm
    '''
    return step_five(step_four(step_three(step_two(step_one(word)))))
