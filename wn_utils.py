
def candidate_selection(wn,
                        token,
                        target_lemma,
                        pos,
                        gold_lexkeys=set(),
                        debug=False):
    """
    return candidate synsets of a token

    :param str token: the token
    :param str targe_lemma: a lemma
    :param str pos: supported: n, v, r, a
    :param str gold_lexkeys: {'congress%1:14:00::'}

    :rtype: tuple
    :return: (candidate_synsets, 
              gold_in_candidates)
    """

    if token.istitle():
        candidate_synsets = wn.synsets(token, pos)

        if not candidate_synsets:
            candidate_synsets = wn.synsets(target_lemma, pos)

    else:
        candidate_synsets = wn.synsets(target_lemma, pos)

    gold_in_candidates = False

    for synset in candidate_synsets:
        # check if gold in candidate
        lexkeys = {lemma.key() for lemma in synset.lemmas()}
        if any(gold_key in lexkeys
               for gold_key in gold_lexkeys):
            gold_in_candidates = True

    return candidate_synsets, gold_in_candidates

def synset2identifier(synset, wn_version):
    """
    return synset identifier of
    nltk.corpus.reader.wordnet.Synset instance

    :param nltk.corpus.reader.wordnet.Synset synset: a wordnet synset
    :param str wn_version: supported: '171 | 21 | 30'

    :rtype: str
    :return: eng-VERSION-OFFSET-POS (n | v | r | a)
    e.g.
    """
    offset = str(synset.offset())
    offset_8_char = offset.zfill(8)

    pos = synset.pos()
    if pos in {'j', 's'}:
        pos = 'a'

    identifier = 'eng-{wn_version}-{offset_8_char}-{pos}'.format_map(locals())

    return identifier


def get_synset2sensekeys(wn, candidates, wn_version, target_lemma, pos, debug=False):
    """
    obtain

    :rtype: tuple
    :return: (list of sensekeys, synset2sensekey)

    """
    sensekeys = []

    synset2sensekeys = dict()

    for synset in candidates:
        sy_id = synset2identifier(synset, wn_version)

        key = None
        strategy = None

        for lemma in synset.lemmas():
            if lemma.name() == target_lemma:
                strategy = 'lemma match'
                key = lemma.key()

            elif lemma.name().lower() == target_lemma.lower():
                strategy = 'lower case'
                key = lemma.key()

        if not key:
            for lemma in synset.lemmas():
                if target_lemma.startswith(lemma.name()):
                    strategy = 'target lemma starts with lemma'
                    key = lemma.key()


        if not key:
            print()
            print('no sensekey found for %s %s wn %s sy_id %s' % (target_lemma,
                                                                  pos,
                                                                  wn_version,
                                                                  sy_id))
            print('falling back on picking first sensekey from first lemma in synset.lemmas()')

        if not key:
            for lemma in synset.lemmas():
                strategy = 'pick first key'
                key = lemma.key()
                break 


        sensekeys.append(key)
        synset2sensekeys[sy_id] = key

    return sensekeys, synset2sensekeys
