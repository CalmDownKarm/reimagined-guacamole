from utils import read_sst5, create_dataloader
from TextProcessing import TextProcessor
from pytorch_transformers import BertTokenizer
import torch.optim as optim
from model import CNN1d
import torch
import torch.nn as nn
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
TEXT_COL, LABEL_COL = 'text', 'sentiment'
datasets = read_sst5("GoogleDrive/My Drive/SST5")

labels = list(set(datasets["train_nl"][LABEL_COL].tolist()))
num_classes = max(labels)
# labels to integers mapping (In case the labels are string)
label2int = {label: i for i, label in enumerate(labels)}


processor = TextProcessor(tokenizer=tokenizer, label2id=label2int, max_length=256)
train_dataloader = processor.create_dataloader(datasets["train_nl"])
test_dataloader = processor.create_dataloader(datasets["test"])
dev_dataloader = processor.create_dataloader(datasets["dev"])

model = CNN1d()

optimizer = optim.Adam(model.parameters(), lr=1e-07)
criterion_neg = nn.NLLLoss()
evaluator = nn.NLLLoss()

criterion_neg = criterion_neg.cuda()
evaluator = evaluator.cuda()

model = model.cuda()

total_epochs = 200


def choose_complement_targets(targets, num_classes):
    complement_targets = (targets + torch.LongTensor(targets.size(0)).random_(1, num_classes)) % num_classes
    return complement_targets


def train_pl(train_dataloader, model, criterion, evaluator, optimizer): #X is train_nl set, Y is validation set, data is the whole data
    model.train();
    total_loss = 0;
    n_samples = 0;
    for i, (inputs, targets) in enumerate(train_dataloader):
        # inputs is of shape [batch_size, sentence_dimension]
        # target is of dimension [batch_size]
        optimizer.zero_grad()
        positive_probabilities = model(inputs)
        loss = criterion(positive_probabilities.cuda(), targets.cuda())
        loss.backward()
        total_loss += loss.data;
        n_samples += inputs.size(0)
        optimizer.step()
    return total_loss/n_samples


def train_nl(train_dataloader, model, criterion_neg, evaluator, optimizer): #X is train_nl set, Y is validation set, data is the whole data
    model.train();
    total_neg_loss = 0;
    total_pos_loss = 0
    n_samples = 0;
    for i, (inputs, targets) in enumerate(train_dataloader):
        # inputs is of shape [batch_size, sentence_dimension]
        # target is of dimension [batch_size]
        optimizer.zero_grad()
        complement_targets = choose_complement_targets(targets, num_classes)
        positive_probabilities = model(inputs)
        complement_probabilities = 1. - positive_probabilities
        neg_loss = criterion_neg(complement_probabilities.cuda(), complement_targets.cuda())
        pos_loss = evaluator(positive_probabilities.cuda(), targets.cuda())
        neg_loss.backward()
        total_neg_loss += neg_loss.data;
        total_pos_loss += pos_loss
        n_samples += inputs.size(0)
        optimizer.step()
    return total_neg_loss/n_samples, total_pos_loss/n_samples


def evaluate_pl(dev_dataloader, model, evaluator): #X is train_nl set, Y is validation set, data is the whole data
    model.eval();
    total_loss = 0;
    n_samples = 0;
    for i, (inputs, targets) in enumerate(dev_dataloader):
        positive_probabilities = model(inputs)
        loss = evaluator(positive_probabilities.cuda(), targets.cuda())
        total_loss += loss.data;
        n_samples += inputs.size(0)
    return total_loss/n_samples


def evaluate_nl(dev_dataloader, model, evaluator): #X is train_nl set, Y is validation set, data is the whole data
    model.eval();
    total_loss = 0;
    n_samples = 0;
    for i, (inputs, targets) in enumerate(dev_dataloader):
        complement_targets = choose_complement_targets(targets, num_classes)
        positive_probabilities = model(inputs)
        complement_probabilities = 1. - positive_probabilities
        loss = evaluator(complement_probabilities.cuda(), complement_targets.cuda())
        total_loss += loss.data;
        n_samples += inputs.size(0)
    return total_loss/n_samples


def threshold_data(train_dataloader, model, threshold):
    model.eval()
    thresholded_inputs = None
    thresholded_targets = None
    for i, (inputs, targets) in enumerate(train_dataloader):
        positive_probabilities = model(inputs)
        indices = positive_probabilities > threshold
        """
        Try Catch block written to handle None values, check using test cases
        """
        try:
            thresholded_inputs = torch.cat((thresholded_inputs, inputs[indices]))
        except:
            thresholded_inputs = inputs[indices]
        """
           check here
        """
        try:
            thresholded_targets = torch.cat((thresholded_targets, targets[indices]))
        except:
            thresholded_targets = targets[indices]

    """
       Using thresholded_inputs and thresholded_targets, create a dataloader
    """
    return create_dataloader(features=thresholded_inputs, labels=thresholded_targets)





"""
    Step 1. Train for NL
"""
for epoch in range(1, total_epochs):
    epoch_start_time = time.time()
    neg_train_loss, pos_train_loss = train_nl(train_dataloader, model, criterion_neg, evaluator, optimizer)
    neg_eval_loss = evaluate_nl(dev_dataloader, model, evaluator)
    if epoch % 5 == 0:
        print('| end of epoch {:3d} | time: {:5.2f}s | negative_train_loss {:5.4f} | positive_train_loss {:5.4f} | valid_neg_loss  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), neg_train_loss, pos_train_loss, neg_eval_loss))


""" 
   Step 2. Filter out the samples which have probability less than 1/c
   Below block contains code for SelNL
"""
threshold = 1/float(num_classes)
for epoch in range(1, total_epochs):
    epoch_start_time = time.time()
    thresholded_data_loader = threshold_data(train_dataloader, model, threshold)
    neg_train_loss, pos_train_loss = train_nl(thresholded_data_loader, model, criterion_neg, evaluator, optimizer)
    neg_eval_loss = evaluate_nl(dev_dataloader, model, evaluator)
    if epoch % 5 == 0:
        print('| end of epoch {:3d} | time: {:5.2f}s | negative_train_loss {:5.4f} | positive_train_loss {:5.4f} | valid_neg_loss  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), neg_train_loss, pos_train_loss, neg_eval_loss))


"""
   Step 3. The below block contains code for SelPL
"""
gamma = float(0.5)
threshold = 1/float(gamma)
for epoch in range(1, total_epochs):
    epoch_start_time = time.time()
    thresholded_data_loader = threshold_data(train_dataloader, model, threshold)
    pos_train_loss = train_pl(thresholded_data_loader, model, criterion_neg, evaluator, optimizer)
    pos_eval_loss = evaluate_pl(dev_dataloader, model, evaluator)
    if epoch % 5 == 0:
        print('| end of epoch {:3d} | time: {:5.2f}s | positive_train_loss {:5.4f} | valid_pos_loss  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), pos_train_loss, pos_eval_loss))

