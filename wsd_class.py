from lstm_utils import load_tensors, ctx_embd_input
import tensorflow as tf
from datetime import datetime
from itertools import islice
import numpy as np


class WsdLstm:
    """

    :ivar str model_path: path to lstm model
    :ivar str vocab_path: path to vocabulary file
    :ivar int debug_value: verbosity of debugging info
    """

    def __init__(self,
                 model_path,
                 vocab_path,
                 debug_value=0):
        self.debug_value = debug_value
        self.sess = tf.InteractiveSession()

        self.vocab = self.load_vocab(vocab_path)

        self.x, \
        self.predicted_context_embs, \
        self.lens = self.load_model(model_path)

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

    def load_model(self, model_path):
        """
        load lstm model

        :param str model_path: path to model

        :rtype: tuple (all of type tensorflow.python.framework.ops.Tensor)
        :return: (x, predicted_context_embs, lens)
        """
        if self.debug_value >= 1:
            print(datetime.now(), 'loading model {model_to_use} from {model_path}'.format_map(locals()))

        saver = tf.train.import_meta_graph(model_path + '.meta', clear_devices=True)
        saver.restore(self.sess, model_path)
        x, predicted_context_embs, lens = load_tensors(self.sess)

        if self.debug_value >= 1:
            print(datetime.now(), 'loaded model {model_to_use} from {model_path}'.format_map(locals()))

        return x, predicted_context_embs, lens

    def apply_model(self, sentences_as_ids, sentence_lens):
        """
        apply

        :rtype: iterable
        :return: iterable of target embeddings
        """
        target_embeddings = self.sess.run(self.predicted_context_embs,
                                          {self.x: sentences_as_ids,
                                           self.lens: sentence_lens})

        return target_embeddings


    def apply_on_lstm_input_file(self, lstm_input_path, batch_size):
        """

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
                    [_list.append(self.vocab['<unkn>']) for _ in range(length_diff)]

                target_embeddings = self.sess.run(self.predicted_context_embs,
                                                  {self.x: annotated_sentences,
                                                   self.lens: sentence_lens})

                for instance_id, target_index, annotation, target_embedding in zip(instance_ids,
                                                                                   target_indices,
                                                                                   identifiers,
                                                                                   target_embeddings):
                    yield (instance_id, target_index, annotation, target_embedding)