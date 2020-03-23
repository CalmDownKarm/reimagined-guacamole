from utils import read_sst5, create_dataloader, read_imdb
from create_dataloader.dataloader_type2 import TextProcessor
from utils_load.load_pretrained_embeddings import load_vectors, load_word_idx_maps, load_lookup_table
import torch.optim as optim
import torch
import torch.nn as nn
import time
from nltk.tokenize import word_tokenize
from models.WordCNN import CNN1d


total_epochs = 10000
batch_size = 512
embedding_location = "/media/data_dump_2/Sanyam/crawl-300d-2M.vec"
load_dataset = read_sst5
dataset_location = "/media/data_dump_2/Sanyam/NLNL_Clean/Data/sst5"
tokenizer = word_tokenize
TEXT_COL, LABEL_COL = 'text', 'sentiment'

datasets = load_dataset(dataset_location)
print("*" * 50, " LOADING PRETRAINED EMBEDDING ", "*" * 50)
embedding, vocab = load_vectors(embedding_location)
word2idx, idx2word = load_word_idx_maps(vocab)
embedding_lookup, pretrained_vectors = load_lookup_table(embedding, word2idx)

labels = list(set(datasets["train"][LABEL_COL].tolist()))
num_classes = max(labels)
# labels to integers mapping (In case the labels are string)
label2int = {label: i for i, label in enumerate(labels)}

processor = TextProcessor(tokenizer=tokenizer, label2id=label2int, word2idx=word2idx, idx2word=idx2word, vocab=vocab, max_length=256)

print("*" * 50, " CREATING DATALOADERS ", "*" * 50)
train_dataloader = processor.create_dataloader(datasets["train"], batch_size)
test_dataloader = processor.create_dataloader(datasets["test"], batch_size)
dev_dataloader = processor.create_dataloader(datasets["dev"], batch_size)


model = CNN1d(pre_trained_embedding=pretrained_vectors, n_filters=5, filter_sizes=[2, 3, 4, 5, 6], output_dim=5, dropout=0.1,
            pad_idx=None)

optimizer = optim.Adam(model.parameters(), lr=.001)
criterion_neg = nn.NLLLoss()
evaluator = nn.NLLLoss()

criterion_neg = criterion_neg.cuda()
evaluator = evaluator.cuda()

torch.cuda.set_device(0)

model = model.cuda()


def choose_complement_targets(targets, num_classes):
    complement_targets = (targets + torch.LongTensor(targets.size(0)).random_(1, num_classes)) % num_classes
    return complement_targets


def train_pl(train_dataloader, model, criterion, evaluator,
             optimizer):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_loss = 0;
    n_samples = 0;
    for i, (inputs, targets) in enumerate(train_dataloader):
        # inputs is of shape [batch_size, sentence_dimension]
        # target is of dimension [batch_size]
        model.zero_grad()
        positive_probabilities = model(inputs)
        loss = criterion(positive_probabilities.cuda(), targets.cuda())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        grad_norm = optimizer.step()
        total_loss += loss.data;
        n_samples += inputs.size(0)
    return total_loss / n_samples


def train_nl(train_dataloader, model, criterion_neg, evaluator,
             optimizer):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_neg_loss = 0;
    total_pos_loss = 0
    n_samples = 0;
    for i, (inputs, targets) in enumerate(train_dataloader):
        # inputs is of shape [batch_size, sentence_dimension]
        # target is of dimension [batch_size]
        model.zero_grad()
        complement_targets = choose_complement_targets(targets, num_classes)
        positive_probabilities = model(inputs)
        complement_probabilities = 1. - positive_probabilities
        neg_loss = criterion_neg(complement_probabilities.cuda(), complement_targets.cuda())
        pos_loss = evaluator(positive_probabilities.cuda(), targets.cuda())
        neg_loss.backward()
        optimizer.step()
        total_neg_loss += neg_loss.data;
        total_pos_loss += pos_loss
        n_samples += inputs.size(0)
    return total_neg_loss / n_samples, total_pos_loss / n_samples


def evaluate_pl(dev_dataloader, model, evaluator):  # X is train set, Y is validation set, data is the whole data
    model.eval();
    total_loss = 0;
    n_samples = 0;
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dev_dataloader):
            positive_probabilities = model(inputs)
            loss = evaluator(positive_probabilities.cuda(), targets.cuda())
            total_loss += loss.data;
            n_samples += inputs.size(0)
    return total_loss / n_samples


def evaluate_nl(dev_dataloader, model, evaluator):  # X is train set, Y is validation set, data is the whole data
    model.eval();
    total_loss = 0;
    n_samples = 0;
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dev_dataloader):
            complement_targets = choose_complement_targets(targets, num_classes)
            positive_probabilities = model(inputs)
            complement_probabilities = 1. - positive_probabilities
            loss = evaluator(complement_probabilities.cuda(), complement_targets.cuda())
            total_loss += loss.data;
            n_samples += inputs.size(0)
    return total_loss / n_samples


def test_accuracy(test_dataloader, model):
    model.eval();
    correct = 0
    length = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            complement_targets = choose_complement_targets(targets, num_classes)
            positive_probabilities = model(inputs)
            values, indices = torch.max(positive_probabilities, dim=1)
            correct += (torch.max(positive_probabilities, 1)[1].view(targets.size()).cuda().data == targets.cuda().data).sum() # Works fine
            length += inputs.size(0)
            # correct += (targets.cuda() == indices.cuda()).sum().data
        accuracy = 100. * correct / float(length)
    return accuracy


def threshold_data(train_dataloader, model, threshold):
    model.eval()
    thresholded_inputs = None
    thresholded_targets = None
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(train_dataloader):
            positive_probabilities = model(inputs)
            indices = torch.diag(positive_probabilities[:, targets.numpy()]) > threshold

            if thresholded_inputs is not None:
                thresholded_inputs = torch.cat((thresholded_inputs, inputs[indices]))
            else:
                thresholded_inputs = inputs[indices]

            if thresholded_targets is not None:
                thresholded_targets = torch.cat((thresholded_targets, targets[indices]))
            else:
                thresholded_targets = targets[indices]

    """
       Using thresholded_inputs and thresholded_targets, create a dataloader
    """
    if thresholded_inputs.size(0) == 0:  # takes care of empty batch
        return None
    return create_dataloader(features=thresholded_inputs, labels=thresholded_targets)


"""
    Step 1. Train for NL
"""
# print("*" * 100 + "NL started" + "*" * 100)
# for epoch in range(1, total_epochs):
#     epoch_start_time = time.time()
#     neg_train_loss, pos_train_loss = train_nl(train_dataloader, model, criterion_neg, evaluator, optimizer)
#     neg_eval_loss = evaluate_nl(dev_dataloader, model, evaluator)
#     if epoch % 5 == 0:
#         accuracy = test_accuracy(test_dataloader, model)
#         print(
#             '| end of epoch {:3d} | time: {:5.2f}s | negative_train_loss {:5.4f} | positive_train_loss {:5.4f} | valid_neg_loss  {:5.4f}'.format(
#                 epoch, (time.time() - epoch_start_time), neg_train_loss, pos_train_loss, neg_eval_loss))
#         print('test accuracy is', accuracy)
#
# """
#    Step 2. Filter out the samples which have probability less than 1/c
#    Below block contains code for SelNL
# """
# print("*" * 100 + "SelNL started" + "*" * 100)
# threshold = 1 / float(num_classes)
# for epoch in range(1, total_epochs):
#     epoch_start_time = time.time()
#     thresholded_data_loader = threshold_data(train_dataloader, model, threshold)
#     if thresholded_data_loader is None:  # Takes care of empty batch
#         break
#     neg_train_loss, pos_train_loss = train_nl(thresholded_data_loader, model, criterion_neg, evaluator, optimizer)
#     neg_eval_loss = evaluate_nl(dev_dataloader, model, evaluator)
#     if epoch % 5 == 0:
#         accuracy = test_accuracy(test_dataloader, model)
#         print(
#             '| end of epoch {:3d} | time: {:5.2f}s | negative_train_loss {:5.4f} | positive_train_loss {:5.4f} | valid_neg_loss  {:5.4f}'.format(
#                 epoch, (time.time() - epoch_start_time), neg_train_loss, pos_train_loss, neg_eval_loss))
#         print('test accuracy is', accuracy)

"""
   Step 3. The below block contains code for SelPL
"""
print("*" * 100 + "SelPL started" + "*" * 100)
gamma = float(0.5)
threshold = 1 / float(gamma)
for epoch in range(1, total_epochs):
    epoch_start_time = time.time()
    # thresholded_data_loader = threshold_data(train_dataloader, model, threshold)
    thresholded_data_loader = train_dataloader
    if thresholded_data_loader is None:  # Takes care of empty batch
        break
    pos_train_loss = train_pl(thresholded_data_loader, model, criterion_neg, evaluator, optimizer)
    pos_eval_loss = evaluate_pl(dev_dataloader, model, evaluator)
    if epoch % 5 == 0:
        accuracy = test_accuracy(test_dataloader, model)
        accuracy_train = test_accuracy(train_dataloader, model)
        print('| end of epoch {:3d} | time: {:5.2f}s | positive_train_loss {:5.4f} | valid_pos_loss  {:5.4f}'.format(
            epoch, (time.time() - epoch_start_time), pos_train_loss, pos_eval_loss))
        print('test accuracy is', accuracy)
        print('train accuracy is', accuracy_train)


