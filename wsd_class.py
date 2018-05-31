from lstm_utils import load_tensors, ctx_embd_input
import tensorflow as tf
from datetime import datetime
from itertools import islice
import numpy as np
from scipy import spatial


class WsdLstm:
    """

    :ivar str model_path: path to lstm model
    :ivar str vocab_path: path to vocabulary file
    :ivar tf.Session: tf session object
    :ivar int debug_value: verbosity of debugging info
    """

    def __init__(self,
                 model_path,
                 vocab_path,
                 sess,
                 debug_value=0):
        self.debug_value = debug_value

        self.vocab = self.load_vocab(vocab_path)

        self.x, \
        self.predicted_context_embs, \
        self.lens = self.load_model(model_path, sess)

    def load_vocab(self, vocab_path):
        """
        load vocabulary

        :param str vocab_path: path to vocabulary

        :rtype: dict
        :return: mapping token -> identifier
        """
        if self.debug_value >= 1:
            print(datetime.now(), 'loading vocab {model_to_use} from {path_vocab}'.format_map(locals()))

        vocab = np.load(vocab_path)

        if self.debug_value >= 1:
            print(datetime.now(), 'loaded vocab {model_to_use} from {path_vocab}'.format_map(locals()))
            print('vocabulary size: %s' % len(vocab))

        return vocab

    def load_model(self, model_path, sess):
        """
        load lstm model

        :param str model_path: path to model
        :param tf.Session: tensorflow session

        :rtype: tuple (all of type tensorflow.python.framework.ops.Tensor)
        :return: (x, predicted_context_embs, lens)
        """
        if self.debug_value >= 1:
            print(datetime.now(), 'loading model {model_to_use} from {model_path}'.format_map(locals()))

        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(sess, model_path)
        x, predicted_context_embs, lens = load_tensors(sess)

        if self.debug_value >= 1:
            print(datetime.now(), 'loaded model {model_to_use} from {model_path}'.format_map(locals()))

        return x, predicted_context_embs, lens

    def apply_model(self, sess, sentences_as_ids, sentence_lens):
        """
        apply

        :param tf.Session sess: tensorflow session
        :param list sentences_as_ids: list of list of ids (representing tokens)
        :param list sentence_lens: list of lengths of each sentence in sentences_as_ids

        :rtype: iterable
        :return: iterable of target embeddings
        """
        target_embeddings = sess.run(self.predicted_context_embs,
                                     {self.x: sentences_as_ids,
                                      self.lens: sentence_lens})

        return target_embeddings


    def apply_on_lstm_input_file(self, sess, lstm_input_path, batch_size):
        """

        :param tf.Session: tensorflow session
        :param str lstm_input_path: path of lstm input
        e.g sentences such as
        the man meets---MEANING women
        :param int batch_size: number of sentences to process with lstm per batch

        :rtype: generator
        :return: generator of (instance_id, target_index, annotation, target_embedding)
        """
        counter = 0
        with open(lstm_input_path) as infile:
            for n_lines in iter(lambda: tuple(islice(infile, batch_size)), ()):

                counter += len(n_lines)

                print(counter, datetime.now())

                identifiers = []  # list of meaning identifiers
                instance_ids = []
                annotated_sentences = []
                target_indices = []
                sentence_lens = []  # list of ints

                for line in n_lines:

                    instance_id, sentence = line.strip().split('\t')
                    tokens, annotation_indices = ctx_embd_input(sentence)

                    for index, synset_id in annotation_indices:

                        sentence_as_ids = [self.vocab.get(w) or self.vocab['<unkn>']
                                           for w in tokens]

                        target_id = self.vocab['<target>']
                        sentence_as_ids[index] = target_id


                        # update batch information
                        identifiers.append(synset_id)
                        instance_ids.append(instance_id)
                        annotated_sentences.append(sentence_as_ids)
                        target_indices.append(index)
                        sentence_lens.append(len(sentence_as_ids))

                # compute embeddings for batch
                max_length = max([len(_list) for _list in annotated_sentences])
                for _list in annotated_sentences:
                    length_diff = max_length - len(_list)
                    [_list.append(self.vocab['<pad>'])  # <pad> used to <unkn>
                     for _ in range(length_diff)]

                target_embeddings = sess.run(self.predicted_context_embs,
                                             {self.x: annotated_sentences,
                                             self.lens: sentence_lens})

                for instance_id, target_index, annotation, target_embedding in zip(instance_ids,
                                                                                   target_indices,
                                                                                   identifiers,
                                                                                   target_embeddings):
                    yield (instance_id, target_index, annotation, target_embedding)


    def wsd_on_test_instance(self,
                             sess,
                             sentence_tokens,
                             target_index,
                             candidate_meanings,
                             meaning_embeddings,
                             meaning_instances,
                             method,
                             debug=0):
        """
        perform wsd on test instance from wsd competition

        :param tf.Session sess: tensorflow session
        :param list sentence_tokens: list of my_classes.Token objects
        :param int target_index: index of target token
        :param list candidate_meanings: list of meaning identifiers,
        each being a candidate meaning of the target token
        :param dict meaning_embeddings: mapping meaning identifier -> embedding
        :param dict meaning_instances: mapping meaning -> list of (instance_id, target_index, target_embedding)
        :param str method: 'averaging' | 'most_similar_instance'


        :param int debug: debug level

        :rtype: tuple
        :return: (wsd_strategy, chosen_meaning, meaning2cosine)
        """
        sentence_as_ids = [self.vocab.get(token_obj.text) or self.vocab['<unkn>']
                           for token_obj in sentence_tokens]
        sentence_as_ids[target_index] = self.vocab['<target>']

        target_embeddings = sess.run(self.predicted_context_embs,
                                     {self.x: [sentence_as_ids],
                                      self.lens: [len(sentence_as_ids)]})

        target_embedding = target_embeddings[0]


        highest_meanings = []
        highest_conf = -100
        meaning2confidence = dict()

        if debug >= 2:
            print()
            print('##')

        for meaning_id in candidate_meanings:
            if meaning_id in meaning_embeddings:

                if method == 'averaging':
                    identifier_embedding = [(meaning_id, meaning_embeddings[meaning_id])]
                elif method == 'most_similar_instance':
                    identifier_embedding = [((instance_id, index_), embedding)
                                             for (instance_id, index_, embedding) in meaning_instances[meaning_id]]

                for id_, cand_embedding in identifier_embedding:

                    sim = 1 - spatial.distance.cosine(cand_embedding, target_embedding)

                    if sim == highest_conf:
                        highest_meanings.append(meaning_id)
                    elif sim > highest_conf:
                        highest_meanings = [meaning_id]
                        highest_conf = sim

                    meaning2confidence[id_] = sim

            else:
                if debug >= 2:
                    print('there is no embedding for', meaning_id)

                meaning2confidence[meaning_id] = 0.0

        wsd_strategy = 'lstm'
        if len(highest_meanings) >= 1:
            highest_meaning = highest_meanings[0]
        else:
            highest_meaning = candidate_meanings[0]
            wsd_strategy = 'mfs_fallback'

        if len(candidate_meanings) == 1:
            wsd_strategy = 'monosemous'


        if debug >= 2:
            rank = candidate_meanings.index(highest_meaning)
            print()
            print(candidate_meanings)
            print(rank, wsd_strategy)

        return (wsd_strategy, highest_meaning, meaning2confidence)
