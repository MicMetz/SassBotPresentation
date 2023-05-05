import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import preprocessing
import models as modeling
from barbar import Bar
import random
import focal_loss
import os


from sklearn.metrics import  f1_score, classification_report
import utils
from transformers import AdamW, get_linear_schedule_with_warmup

# tries to run code off of graphics card, otherwise uses a cpu instead.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# directory to save checkpoints such as best model under.
ckpt_dir = './ckpts'

# main training loop
def train(base_model, mt_classifier, iterator, optimizer, sar_criterion, scheduler):
    # set the model in eval phase
    base_model.train(True)
    mt_classifier.train(True)
    acc_sarcasm= 0
    loss_sarc= 0

    # extracts each bit of data and its label.
    for data_input, label_input  in Bar(iterator):

        # mounts data onto the device.
        for k, v in data_input.items():
            data_input[k] = v.to(device)

        # mounts labels onto the device.
        for k, v in label_input.items():
            label_input[k] = v.to(device)

        optimizer.zero_grad()


        #forward pass

        sarcasm_target = label_input['sarcasm']

        # forward pass

        output = base_model(**data_input)
        sarcasm_logits = mt_classifier(output)


        # compute the loss
        loss_sarcasm = sar_criterion(sarcasm_logits.squeeze(), sarcasm_target)
        loss_sarc += loss_sarcasm.item()

        # backpropage the loss and compute the gradients
        loss_sarcasm.backward()
        optimizer.step()
        scheduler.step()
        sarcasm_probs = torch.sigmoid(sarcasm_logits)
        acc_sarcasm += utils.binary_accuracy2(sarcasm_probs, sarcasm_target)

    # stores and returns accuraces and loss.
    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator)}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses

# main validation loop.
def evaluate(base_model, mt_classifier, iterator, sar_criterion):
    # initialize every epoch
    acc_sarcasm= 0
    loss_sarc= 0

    # used to store outputs and labels for f1 score calculations later down the line.
    all_sarcasm_outputs = []
    all_sarcasm_labels = []

    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()
    with torch.no_grad():
        # once again runs off of all the data.
        for data_input, label_input in Bar(iterator):

            # mounts input onto device.
            for k, v in data_input.items():
                data_input[k] = v.to(device)

            # mounts labels onto device.
            for k, v in label_input.items():
                label_input[k] = v.to(device)



            sarcasm_target = label_input['sarcasm']

            # forward pass

            output = base_model(**data_input)
            sarcasm_logits = mt_classifier(output)

            sarcasm_probs = torch.sigmoid(sarcasm_logits).to(device)

            # compute the loss
            loss_sarcasm = sar_criterion(sarcasm_logits.squeeze(), sarcasm_target)

            # compute the running accuracy and losses
            acc_sarcasm += utils.binary_accuracy2(sarcasm_probs, sarcasm_target)

            # summed loss for average loss calculation later.
            loss_sarc += loss_sarcasm.item()

            # converts everything into a probability and then converts it into a label.
            predicted_sarcasm = torch.round(sarcasm_probs)
            all_sarcasm_outputs.extend(predicted_sarcasm.squeeze().int().cpu().numpy())
            all_sarcasm_labels.extend(sarcasm_target.squeeze().int().cpu().numpy())

    # reports sarcasm and metrics.
    fscore_sarcasm = f1_score(y_true=all_sarcasm_labels,y_pred=all_sarcasm_outputs , average='macro')
    report_sarcasm = classification_report(y_true=all_sarcasm_labels, y_pred=all_sarcasm_outputs,digits=4)


    accuracies = { 'Sarcasm': acc_sarcasm / len(iterator), 'F1_sarcasm': fscore_sarcasm, 'Report_sarcasm': report_sarcasm}
    losses = { 'Sarcasm': loss_sarc / len(iterator)}
    return accuracies, losses

# predicts using model with mt_classifier with iterator as input.
def predict(base_model, mt_classifier, iterator):
    # initialize every epoch

    # added output saving option for prediction f1 score calculations later.
    all_outputs = []
    # set the model in eval phase
    base_model.eval()
    mt_classifier.eval()


    with torch.no_grad():
        for data_input in Bar(iterator):

            for k, v in data_input.items():
                data_input[k] = v.to(device)

            # forward pass

            output = base_model(**data_input)

            logits = mt_classifier(output)

            probs = torch.sigmoid(logits).to(device)
            # compute the loss
            predicted = torch.round(probs)
            all_outputs.extend(predicted.squeeze().int().cpu().numpy())

    return  all_outputs

# evaulates off of the best model it has.
def eval_full(config, loader1, test_labels):
    # loads data by criterion
    criterion = config['loss']
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    classifier = modeling.ATTClassifier(base_model.output_num(), class_num=1).to(device)
    base_model.load_state_dict(torch.load(f"./ckpts/best_basemodel_sarcasm_{config['args'].lm_pretrained}_{criterion}_sar.pth"))
    classifier.load_state_dict(torch.load(f"./ckpts/best_cls_sarcasm_{config['args'].lm_pretrained}_{criterion}_sar.pth"))
    # loads model and classifier onto device.
    base_model = base_model.to(device)
    classifier = classifier.to(device)
    df = pd.DataFrame()
    col = 'task_a_en'

    # predicts
    all_outputs = predict(base_model, classifier, loader1)
    df[col] = all_outputs

    # reports f1 score and other relevant summary statistics.
    print(classification_report(y_true=test_labels,y_pred=all_outputs , digits=4))

    # saves
    df.to_csv(f'results/{col}_{criterion}_sar.txt', index=False, header=True)

# config is the hyper parameters, train_loader is the training data.
def train_full(config, train_loader, valid_loader):

    #Instanciate models
    base_model = modeling.TransformerLayer(pretrained_path=config['pretrained_path'], both=True).to(device)
    mtl_classifier = modeling.ATTClassifier(base_model.output_num(), class_num=1).to(device)
    criterion = config['loss']


    ## set optimizer and criterions
    if criterion =='WBCE':
        sarc_criterion = focal_loss.WeightedBCELoss().to(device)
    elif criterion =='FL':
        sarc_criterion = focal_loss.BinaryFocalLoss(alpha=0.6).to(device)
    else:
        sarc_criterion = nn.BCEWithLogitsLoss().to(device)

    # oraganizes parameters for us.
    params = [{'params':base_model.parameters(), 'lr':config['lr']}, {'params': mtl_classifier.parameters(), 'lr': config['lr']}]#, {'params':multi_task_loss.parameters(), 'lr': 0.0005}]
    optimizer = AdamW(params, lr=config["lr"])
    train_data_size = len(train_loader)
    num_train_steps = len(train_loader) * config['epochs']
    warmup_steps = int(config['epochs'] * train_data_size * 0.1 / config['batch_size'])
    scheduler = get_linear_schedule_with_warmup(optimizer,  num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    # Train model
    best_sarcasm_valid_accuracy = 0
    best_total_val_acc = 0
    report_sarcasm = None
    epo = 0

    #trains across epochs to see if it can train something better.
    for epoch in range(config['epochs']):
        print("epoch {}".format(epoch + 1))

        train_accuracies, train_losses = train(base_model, mtl_classifier, train_loader, optimizer, sarc_criterion,scheduler)
        valid_accuracies, valid_losses = evaluate(base_model, mtl_classifier, valid_loader, sarc_criterion)
        total_val_acc = valid_accuracies['F1_sarcasm']

        # saves finding if it turns out to be the best possible option.
        if total_val_acc > best_total_val_acc:
            epo = epoch
            best_total_val_acc = total_val_acc
            best_sarcasm_valid_accuracy = valid_accuracies['F1_sarcasm']
            report_sarcasm = valid_accuracies['Report_sarcasm']
            best_sarcasm_loss = valid_losses['Sarcasm']
            print("save model's checkpoint")

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)

            torch.save(base_model.state_dict(), f"./ckpts/best_basemodel_sarcasm_{config['args'].lm_pretrained}_{criterion}_sar.pth")
            torch.save(mtl_classifier.state_dict(), f"./ckpts/best_cls_sarcasm_{config['args'].lm_pretrained}_{criterion}_sar.pth")

        # prints findings.
        print('********************Train Epoch***********************\n')
        print("accuracies**********")
        for k , v in train_accuracies.items():
            print(k+f" : {v * 100:.2f}")
        print("losses**********")
        for k , v in train_losses.items():
            print(k+f": {v :.5f}\t")
        print('********************Validation***********************\n')
        print("accuracies**********")
        print(valid_accuracies['Report_sarcasm'])
        print("losses**********")
        for k, v in valid_losses.items():
            print(k + f": {v :.5f}\t")
        print('******************************************************\n')
    # reports the best possible findings so far.
    print(f"epoch of best results {epo}")
    with open(f'report_Sarcasm_{config["args"].lm_pretrained}_{criterion}.txt', 'w') as f:
        f.write("Sarcasm report\n")
        f.write(report_sarcasm)
    return best_sarcasm_valid_accuracy, best_sarcasm_loss


if __name__ == "__main__":
    # checks for funny input and shuts it down if there is.
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')


    # loads arguments.
    parser = argparse.ArgumentParser(description='Conditional Domain Adversarial Network')
    parser.add_argument('--lm_pretrained', type=str, default='xlml',
                        help=" path of pretrained transformer")
    parser.add_argument('--lr', type=float, default=2e-5, help="learning rate")

    parser.add_argument('--batch_size', type=int, default=36, help="training batch size")
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--loss', type=str, default="WBCE", choices=['WBCE', 'FL', 'BCE'])

    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 10)')
    args = parser.parse_args()


    config = {}
    config['args'] = args
    config["output_for_test"] = True
    config['epochs'] = args.epochs
    config["class_num"] = 1
    config["lr"] = args.lr
    config['batch_size'] = args.batch_size
    config['lm'] = args.lm_pretrained
    config['loss'] = args.loss

    # loads model types.
    dosegmentation = False
    if args.lm_pretrained == 'xlml':
        config['pretrained_path'] = "xlm-roberta-large"
    elif args.lm_pretrained == 'xlm': # Pay attention to here! python main.py --lm_pretrained xlm --lr 2e-5 --batch_size 20 --num_worker 4 --loss WBCE --phase train --epochs 1
        config['pretrained_path'] = "xlm-roberta-base"
        dosegmentation = True
    else:
        print("not supported model")


    # used to control random choices and stuff.
    RANDOM_SEED = 3407#, 12346, 12347, 12348, 12349]

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # preforms train or predict tasks.
    if args.phase == 'train':
        train_loader, valid_loader = preprocessing.loadTrainValData2(batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        best_sarcasm_acc, best_sarcasm_loss =train_full(config, train_loader, valid_loader)
        print(f'  Val. Sarcasm F1: {best_sarcasm_acc * 100:.2f}%  \t Val Sarcasm Loss {best_sarcasm_loss :.4f} ')
    elif args.phase == 'predict':
        test_loader, test_labels = preprocessing.loadTestData(batchsize=args.batch_size, num_worker=0, pretraine_path=config['pretrained_path'])
        eval_full(config, loader1=test_loader, test_labels = test_labels)

    else:
        print("I'm sorry but you did not pick a valid task, please choose 'train' or 'predict' for phase.")
