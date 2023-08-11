import re
from collections import defaultdict

class BPE:
    def vocab_freq(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out
    
    def gen_vocab(self, data):
        vocab = defaultdict(int)
        for line in data:
            for word in line.split():
                vocab[' '.join(list(word)) + ' </w>'] += 1
        return vocab
    
    def byte_pair_encoding(self, data, n):
        vocab = self.gen_vocab(data)
        for i in range(n):
            pairs = self.vocab_freq(vocab)
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
        return vocab


if __name__ == "__main__":
    data = ["hello world", "test", "byte pair encoding"]
    bpe = BPE()
    merged_pairs = bpe.byte_pair_encoding(data, 10)
    print(merged_pairs)