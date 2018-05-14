

def load_tensors(sess):
    x = sess.graph.get_tensor_by_name('Model_1/x:0')
    predicted_context_embs = sess.graph.get_tensor_by_name('Model_1/predicted_context_embs:0')
    lens = sess.graph.get_tensor_by_name('Model_1/lens:0')

    return x, predicted_context_embs, lens


def ctx_embd_input(sentence):
    """
    given a annotated sentence, return
    each the sentence with only one annotation

    :param str sentence: a sentence with annotations
    (lemma---annotation)

    :rtype: generator
    :return: generator of input for the lstm (synset_id, sentence)
    """
    sent_split = sentence.split()

    annotation_indices = []
    tokens = []
    for index, token in enumerate(sent_split):
        token, *annotation = token.rsplit('---')
        tokens.append(token)

        if annotation:
            annotation_indices.append((index, annotation[0]))

    return tokens, annotation_indices
