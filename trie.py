from tqdm import tqdm
import torch


class Trie:
    def __init__(self, tokens_lists, empty_token=0):
        self.root = {}
        self.empty_token = empty_token
        self.nodes_count = 0
        self.total_paths = 0  # Total number of paths (sequences) in the trie
        self.path_counts = {}  # Will store the number of paths through each node

        pbar = tqdm(tokens_lists)
        for word_list in pbar:
            self._insert(word_list)
            self.total_paths += 1  # Increment path count for each sequence
            pbar.set_description(f"Number of nodes in the trie: {self.nodes_count}")

        # Calculate path counts for each node
        self._calculate_path_counts()

    def _insert(self, word_list):
        current = self.root
        for num in word_list:
            if num not in current:
                current[num] = {}
                self.nodes_count += 1
            current = current[num]

    def _calculate_path_counts(self):
        """Calculate how many paths go through each node in the trie."""

        def _count_paths(node, prefix=()):
            # Base case: leaf node
            if not node:
                return 1

            # Path count for this node
            count = 0

            for token, child in node.items():
                current_prefix = prefix + (token,)
                child_count = _count_paths(child, current_prefix)
                count += child_count

                # Store path count for this node
                self.path_counts[current_prefix] = child_count

            return count

        _count_paths(self.root)

    def get_path_reduction(self, sequence):
        """
        Calculate path reduction ratio for each token in a sequence.

        Args:
            sequence: List of token indices

        Returns:
            List of ratios [parent_paths / total_paths] for each token
        """
        ratios = []
        current_prefix = ()
        parent_paths = self.total_paths

        for token in sequence:
            # Update the current prefix
            current_prefix = current_prefix + (token,)

            # Get the number of paths through this node
            current_paths = self.path_counts.get(current_prefix, 0)

            # Calculate the reduction ratio
            reduction_ratio = (parent_paths - current_paths) / (
                    self.total_paths - 1) if parent_paths > 0 else 0  # - 1 is for the current path
            ratios.append(reduction_ratio)

            # Update parent_paths for the next token
            parent_paths = current_paths

        return ratios

    def search_prefix(self, prefix):
        current = self.root
        for num in prefix:
            if num not in current:
                return [self.empty_token]
            current = current[num]
        if len(current) == 0:
            return [self.empty_token]
        return list(current.keys())

    def print_trie(self, tokenizer=None):
        def print_node(node, depth=0):
            for key, value in node.items():
                if tokenizer is not None:
                    print("  " * depth + tokenizer.decode([key]))
                else:
                    print("  " * depth + str(key))
                print_node(value, depth + 1)

        print_node(self.root)


def build_trie(word_list, tokenizer, max_length=512):
    tokens_list = []
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.vocab.get("<eos>")
    for word in word_list:
        tokens = tokenizer.encode(word, add_special_tokens=True)

        if len(tokens) > max_length:
            tokens = tokens[:max_length - 1] + [eos]
        tokens_list.append(tokens)
    return Trie(tokens_list, empty_token=tokenizer.pad_token_id)


def build_mask_from_trie(trie, sequences, vocab_size, return_path_weights=False):
    """
    Generate a mask tensor indicating valid next tokens based on a trie and input sequences.

    Args:
        trie (Trie): The trie structure where each node represents a token and contains child nodes.
        sequences (torch.Tensor): Tensor of shape (batch_size, seq_length) containing token indices.
        vocab_size (int): The size of the vocabulary.

    Returns:
        torch.Tensor: A mask tensor of shape (batch_size, seq_length, vocab_size) with 1s indicating valid next tokens.
        torch.Tensor: A weight tensor of shape (batch_size, seq_length) with path reduction weights.
    """
    batch_size, seq_length = sequences.shape
    mask = torch.zeros((batch_size, seq_length, vocab_size), dtype=torch.float)
    # Add a tensor to store path reduction weights
    path_weights = torch.zeros((batch_size, seq_length), dtype=torch.float)

    for batch_idx in range(batch_size):
        # Convert sequence to list for path reduction calculation
        sequence = sequences[batch_idx].tolist()

        # Calculate path reduction weights for this sequence
        reduction_ratios = trie.get_path_reduction(sequence)

        # Fill in the path weights tensor
        for i, ratio in enumerate(reduction_ratios):
            if i < seq_length:
                path_weights[batch_idx, i] = ratio

        # Original mask construction
        current_node = trie.root
        for seq_idx in range(seq_length):
            token = int(sequences[batch_idx, seq_idx].item())
            if token == trie.empty_token:
                break
            for child_token in current_node[token]:
                mask[batch_idx, seq_idx, child_token] = 1
            current_node = current_node[token]
    if not return_path_weights:
        return mask
    return mask, path_weights


if __name__ == "__main__":
    # Example usage
    from train import QuantizeTokenizer

    esm_tokenizer = QuantizeTokenizer()
    with open("data/drugbank/train_enzyme_q.txt", "r") as f:
        trie_files = f.read().splitlines()
    trie = build_trie(list(set(trie_files)), esm_tokenizer, max_length=512)
    print("Trie structure:")
    exp = "1 10 0 14 9 14 0 5 2 10 13 2 12 5 5 10 11 0 0 0 0 6 0 0 0 7 0 0 0 0 3 0 0 0 0 5 0 0 13 3 0 0 13 5 0 7 12 0 0 0 0 0 1 8 0 0 0 0 13 0 7 10 0 0 0 0 0 0 13 3 13 2 0 0 0 0 0 11 0 8 0 0 0 0 0 9 9 14 14 1 0 0 0 0 0 0 3 0 0 0 0 0 9 4 8 0 0 0 0 13 0 0 5 9 0 0 9 0 14 9 0 0 3 2".split()
    exp = [int(float(x)) for x in exp]
    exp = [esm_tokenizer.bos_token_id] + exp + [esm_tokenizer.pad_token_id]
    mask, path_weights = build_mask_from_trie(trie, torch.tensor(exp).unsqueeze(0), vocab_size=esm_tokenizer.vocab_size)
    print("Mask shape:", mask.shape)
    print("Path weights shape:", path_weights.shape)
    print("Mask:", mask)
    print("Path weights:", path_weights, path_weights.sum(dim=1))
