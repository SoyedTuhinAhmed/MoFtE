import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets

def get_dataset_fn(dataset_name):
    """
    Given a dataset name e.g. "wikitext-2", convert it into a function name 
    (e.g. "WikiText2") and try to retrieve it from torchtext.datasets.
    """
    # Remove non-alphanumeric characters and capitalize each word.
    # Example: "wikitext-2" -> "Wikitext2"
    func_name = ''.join(word.capitalize() for word in dataset_name.split('-'))
    try:
        dataset_fn = getattr(torchtext.datasets, func_name)
    except AttributeError:
        raise ValueError(f"Dataset function for '{dataset_name}' not found in torchtext.datasets.")
    return dataset_fn

class TextLMDataLoader:
    def __init__(self, dataset_name="wikitext-2", batch_size=32, tokenizer_name="basic_english", specials=["<unk>"]):
        """
        Args:
            dataset_name (str): Name of the dataset as available in torchtext.datasets (e.g. "wikitext-2").
            batch_size (int): Number of sequences per batch.
            tokenizer_name (str): Name of the tokenizer to use.
            specials (list): Special tokens to add to the vocabulary.
        """
        self.batch_size = batch_size
        self.tokenizer = get_tokenizer(tokenizer_name)
        # Dynamically obtain the dataset function (e.g., torchtext.datasets.WikiText2)
        self.dataset_fn = get_dataset_fn(dataset_name)
        
        # Build vocabulary from the training split.
        train_iter = self.dataset_fn(split='train')
        self.vocab = build_vocab_from_iterator(self._yield_tokens(train_iter), specials=specials)
        self.vocab.set_default_index(self.vocab["<unk>"])

    def _yield_tokens(self, data_iter):
        """
        Yield tokens from each line in the dataset.
        """
        for text in data_iter:
            yield self.tokenizer(text)

    def _data_process(self, raw_text_iter):
        """
        Convert each line of text to a tensor of token indices and then concatenate them.
        """
        data = [torch.tensor(self.vocab(self.tokenizer(item)), dtype=torch.long)
                for item in raw_text_iter if item.strip()]
        return torch.cat(tuple(data))

    def _batchify(self, data):
        """
        Ensure data can be evenly divided into batch_size parts and reshape into (num_batches, batch_size).
        """
        n_batches = data.size(0) // self.batch_size
        data = data.narrow(0, 0, n_batches * self.batch_size)
        # Reshape data into (batch_size, -1) then transpose to (num_batches, batch_size)
        return data.view(self.batch_size, -1).t().contiguous()

    def get_data(self):
        """
        Loads and processes the dataset, then returns train, validation, and test batches.
        
        Returns:
            tuple: (train_batches, valid_batches, test_batches) as tensors.
        """
        # Get the three splits from the dataset function.
        train_iter, valid_iter, test_iter = self.dataset_fn()
        
        train_data = self._data_process(train_iter)
        valid_data = self._data_process(valid_iter)
        test_data  = self._data_process(test_iter)
        
        train_batches = self._batchify(train_data)
        valid_batches = self._batchify(valid_data)
        test_batches  = self._batchify(test_data)
        
        return train_batches, valid_batches, test_batches

# Example usage:
if __name__ == '__main__':
    # Specify the dataset name (e.g., "wikitext-2" or "ptb" if available)
    loader = TorchTextLMDataLoader(dataset_name="wikitext-2", batch_size=32)
    train_batches, valid_batches, test_batches = loader.get_data()
    print("Train batches shape:", train_batches.shape)
    print("Validation batches shape:", valid_batches.shape)
    print("Test batches shape:", test_batches.shape)