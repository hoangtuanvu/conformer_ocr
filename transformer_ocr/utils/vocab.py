class VocabBuilder:
    def __init__(self, tokens):
        self.pad = 0
        self.blank = 1
        self.tokens = tokens

        self.tok_2_index = {token: idx + 2 for idx, token in enumerate(tokens)}
        self.index_2_tok = {idx + 2: token for idx, token in enumerate(tokens)}

        # Add <pad> and <blank> token
        self.index_2_tok[0] = '<pad>'
        self.index_2_tok[1] = '<blank>'

    def encode(self, tokens):
        encode_lst = []

        for token in tokens:
            if token in list(self.tok_2_index.keys()):
                encode_lst.append(self.tok_2_index[token])
            else:
                print("{} not in the list!".format(token))
                pass

        return encode_lst

    def decode(self, encoded_sentence):
        last_idx = encoded_sentence.index(self.pad) if self.pad in encoded_sentence else None
        decoded_sent = ''.join([self.index_2_tok[idx] for idx in encoded_sentence[:last_idx]])
        return decoded_sent

    def batch_decode(self, encoded_sentences):
        sentences = [self.decode(encoded_sentence) for encoded_sentence in encoded_sentences]
        return sentences

    def get_vocab_tokens(self):
        vocab_tokens = [self.index_2_tok[idx] for idx in range(len(self.index_2_tok))]

        return vocab_tokens

    def __len__(self):
        return len(self.index_2_tok)


