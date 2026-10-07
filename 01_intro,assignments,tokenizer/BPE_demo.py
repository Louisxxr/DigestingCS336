from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: dict[tuple[int, int], int]):
        self.vocab = vocab
        self.merges = merges

    def encode(self, string: str) -> list[int]:
        """根据 merges 进行编码"""
        tokens = list(map(int, string.encode("utf-8")))

        while True:
            temp_tokens = []
            do_merge = False
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) in self.merges:
                    do_merge = True
                    temp_tokens.append(self.merges[(tokens[i], tokens[i + 1])])
                    i += 1
                else:
                    temp_tokens.append(tokens[i])
                i += 1
            if not do_merge:
                break
            tokens = temp_tokens

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """根据 vocab 进行解码"""
        bytes_seq = b"".join([self.vocab[token] for token in tokens])
        return bytes_seq.decode("utf-8")


def train_bpe(string: str, num_merges: int) -> BPETokenizer:
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    merges: dict[tuple[int, int], int] = {}

    tokens = list(map(int, string.encode("utf-8")))

    for _ in range(num_merges):
        counts = defaultdict(int)  # 创建一个默认值为 0 的整数字典
        for token_1, token_2 in zip(tokens, tokens[1:]):
            counts[(token_1, token_2)] += 1
        if not counts:
            break

        pair = max(counts, key=counts.get)
        token_1, token_2 = pair
        new_token = len(vocab)
        vocab[new_token] = vocab[token_1] + vocab[token_2]
        merges[pair] = new_token

        temp_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                temp_tokens.append(new_token)
                i += 1
            else:
                temp_tokens.append(tokens[i])
            i += 1
        tokens = temp_tokens

    return BPETokenizer(vocab=vocab, merges=merges)


tokenizer = train_bpe(string="the cat in the hat", num_merges=3)

string = "the quick brown fox"
tokens = tokenizer.encode(string)
reconstructed_string = tokenizer.decode(tokens)
assert string == reconstructed_string

print(len(tokenizer.vocab), "tokens in vocab")
print("String:", string)
print("Tokens:", tokens)