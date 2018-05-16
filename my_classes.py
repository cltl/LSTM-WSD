

def generate_training_instances_v2(sentence_tokens,
                                   sentence_lemmas,
                                   sentence_pos,
                                   annotations):
    """
    given the lemmas in a sentence with its annotations (can be more than one)
    generate all training instances for that sentence

    e.g.
    sentence_tokens = ['the', 'man',            'meets',   'women']
    sentence_lemmas = ['the', 'man',            'meet',    'woman']
    sentence_pos    = ['',    'n',              'v',       'n']
    annotations =     [[],    ['1', '2' ],      ['4'],     ['5', '6']]

    would result in
    ('man', 'n', '1', ['the', 'man', 'meets', 'women'], 'the man---1 meets women', 1)
    ('man', 'n', '2', ['the', 'man', 'meets', 'women'], 'the man---2 meets women', 1)
    ('meet', 'v', '4', ['the', 'man', 'meets', 'women'], 'the man meets---4 women', 2)
    ('woman', 'n', '5', ['the', 'man', 'meets', 'women'], 'the man meets women---5', 3)
    ('woman', 'n', '6', ['the', 'man', 'meets', 'women'], 'the man meets women---6', 3)

    :param list sentence_tokens: see above
    :param list sentence_lemmas: see above
    :param list sentence_pos: see above
    :param list annotations: see above

    :rtype: generator
    :return: generator of (target_lemma,
                           target_pos,
                           token_annotation,
                           sentence_tokens,
                           training_example,
                           target_index)
    """
    for target_index, token_annotations in enumerate(annotations):

        target_lemma = sentence_lemmas[target_index]
        target_pos = sentence_pos[target_index]

        for token_annotation in token_annotations:

            if token_annotation is None:
                continue

            a_sentence = []
            for index, token in enumerate(sentence_tokens):

                
                if index == target_index:
                    a_sentence.append(token + '---' + token_annotation)
                else:
                    a_sentence.append(token)

            training_example = ' '.join(a_sentence)

            yield (target_lemma,
                   target_pos,
                   token_annotation,
                   sentence_tokens,
                   training_example,
                   target_index)

treebank_tagset = {
 'CC',
 'CD',
 'DT',
 'EX',
 'FW',
 'IN',
 'JJ',
 'JJR',
 'JJS',
 'LS',
 'MD',
 'NN',
 'NNP',
 'NNPS',
 'NNS',
 'PDT',
 'POS',
 'PRP',
 'PRP$',
 'RB',
 'RBR',
 'RBS',
 'RP',
 'SYM',
 'TO',
 'Tag',
 'UH',
 'VB',
 'VBD',
 'VBG',
 'VBN',
 'VBP',
 'VBZ',
 'WDT',
 'WP',
 'WP$',
 'WRB'}

wordnet_tagset = {'n', 'v', 'r', 'a'}

pwgc_pos2wordnet = {'NN': 'n', 'VB': 'v', 'JJ': 'a', 'R': 'r', 'J' : 'a'}

universal2wordnet = {'NOUN' : 'n',
                     'VERB' : 'v',
                     'ADJ' : 'a',
                     'ADV' : 'r'}

def treebank2wordnet(treebank_pos):
    """
    a treebank pos tag
    (https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)

    :param str treebank_pos: treebank pos tag

    :rtype: str
    :return: n, v, a, r, ''
    """

    assert treebank_pos in treebank_tagset, '{treebank_pos} not in treebank tagset'.format_map(locals())

    wordnet_pos = ''
    if treebank_pos.startswith('NN'):
        wordnet_pos = 'n'
    elif treebank_pos.startswith('JJ'):
        wordnet_pos = 'a'
    elif treebank_pos.startswith('VB'):
        wordnet_pos = 'v'
    elif treebank_pos.startswith('RB'):
        wordnet_pos = 'r'

    return wordnet_pos


failure = False
try:
    treebank2wordnet('sdfs')
except AssertionError:
    failure = True
assert failure

assert treebank2wordnet('NNPS') == 'n'
assert treebank2wordnet('JJS') == 'a'
assert treebank2wordnet('RP') == ''



class Token:
    """
    representation of a token

    :
    """
    def __init__(self,
                token_id,
                text,
                lemma,
                lexkeys=set(),
                synsets=set(),
                treebank_pos=None,
                universal_pos=None,
                pos=None):
        self.token_id = token_id
        self.text = text
        self.lemma = lemma
        self.lexkeys = lexkeys
        self.synsets = synsets

        if treebank_pos:

            if treebank_pos in pwgc_pos2wordnet:
                self.pos = pwgc_pos2wordnet[treebank_pos]
            elif treebank_pos in treebank_tagset:
                self.pos = treebank2wordnet(treebank_pos)
            else:
                self.pos = ''

        if universal_pos:
            if universal_pos in universal2wordnet:
                self.pos = universal2wordnet[universal_pos]
            else:
                self.pos = ''



class Sentence:
    """
    representation of sentence

    :ivar list tokens: list of Ctokens instances
    :ivar str id: instance id of sentence
    """
    def __init__(self, id, tokens):
        self.tokens = tokens
        self.id = id


    def sent_in_lstm_format(self, level, only_keep=set()):
        """
        generate lstm format training examples

        :param str level: sensekey | synset

        see generate_training_instances_v2 for more information

        :rtype: generator
        :return: generator of training examples (annotation, training example)
        """
        sentence_tokens = []
        sentence_lemmas = []
        sentence_pos = []
        annotations = []


        for token in self.tokens:

            sentence_tokens.append(token.text)
            sentence_lemmas.append(token.lemma)
            sentence_pos.append(token.pos)

            if level == 'sensekey':
                if not only_keep:
                    annotations.append(list(token.lexkeys))
                elif only_keep:
                    overlap = token.lexkeys & only_keep
                    if overlap:
                        annotations.append(list(overlap))
                    else:
                        annotations.append([])


            elif level == 'synset':

                if not only_keep:
                    annotations.append(list(token.synsets))
                elif only_keep:
                    overlap = token.synsets & only_keep
                    if overlap:
                        annotations.append(list(overlap))
                    else:
                        annotations.append([])

        for (target_lemma,
             target_pos,
             token_annotation,
             sentence_tokens,
             training_example,
             target_index) in generate_training_instances_v2(sentence_tokens,
                                                             sentence_lemmas,
                                                             sentence_pos,
                                                             annotations):

            yield token_annotation, training_example


    def sentence(self, instance_id):
        """
        print sentence


        :rtype: str
        :return: the sentence
        """
        tokens = []
        for token_obj in self.token_objs:

            if token_obj.instance_id == instance_id:
                tokens.append('***%s***' % token_obj.token)
            else:
                tokens.append(token_obj.token)

        return ' '.join(tokens)
