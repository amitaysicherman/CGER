# load the data/drugbank/test_enzyme.txt
# data/drugbank/test_reaction.txt files
# for postive exapline. load  data/drugbank/DrugBank.txt and ge tall the negative examples.


# load the model and tokenizer

# apply the model to the test set with prediction (teacher forcing) and save the mean log_probability for all the nonmasked tokens

# save the scores in a file

# calcualte the AUC and plot the ROC curve.

import torch

from train import get_encoder_decoder, load_files, SrcTgtDataset, EnzymeDecoder
from trie import build_trie

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_train, tgt_train, src_test, tgt_test = load_files(level="drugbank")
reaction_model, reaction_tokenizer, decoder, esm_tokenizer = get_encoder_decoder(decoder_size="l", dropout=0.2,
                                                                                 drugbank=True)
reaction_model.to(device).eval()
decoder.to(device).eval()

test_dataset = SrcTgtDataset(src_test, tgt_test, reaction_tokenizer, esm_tokenizer, reaction_model)
trie = build_trie(list(set(tgt_train + tgt_test)), esm_tokenizer)
model = EnzymeDecoder(decoder, trie=trie, encoder_dim=768)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {n_params:,}")

model_path = "results/drugbank_l_0.2_0.0001/checkpoint-7000/pytorch_model.bin"
model.load_state_dict(torch.load(model_path))
model.eval().to(device)
