import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import Dataset
from emoji import UNICODE_EMOJI
import re

####### preprocessing #######

# used to determine if character s is an EMOJI or not.
def is_emoji(s):
    return s in UNICODE_EMOJI

# add space near your emoji
def add_space(text):
    return ''.join(' ' + char if is_emoji(char) else char for char in text).strip()

# main preprocessing function.
def preprocess(text):
    # adds space around emojis.
    sent = add_space(text)
    # replaces specific user @'s with just "@user".
    sent = re.sub(r'(?:@[\w_]+)', "@user", sent)
    # replaces urls with just "http".
    sent = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "http", sent)
    # replace underscore with a space.
    sent = sent.replace('_', ' ')
    # replace hastags with spaces.
    sent = sent.replace('#', ' ')

    return sent

def loadTrainValData2(size=0.2, batchsize=16, num_worker=0, pretraine_path="xlm-roberta-base", seed=42):
    # loads the training data.
    path = "data/train.En.csv"

    data = pd.read_csv(path, encoding='utf-8')
    data = data[~data.text.isna()]
    # with preprocessing to data.
    data['text'] = data['text'].apply(lambda x: preprocess(x))
    # creates seperate dataset that is just rephrasing of the original content for extra training data.
    rephrase_df = data[["rephrase"]]
    rephrase_df = rephrase_df.dropna()
    rephrase_df['rephrase'] = rephrase_df['rephrase'].apply(lambda x: preprocess(x))
    print(rephrase_df.shape)
    rephrase_df["sarcastic"] = 0
    rephrase_df.columns = ["text", "sarcastic"]
    # structures the data, and splits it into usable parts.
    data = data[["text", "sarcastic"]]
    data = data.sample(frac=1).reset_index(drop=True)
    df_train, df_test = train_test_split(data, test_size=size, stratify=data["sarcastic"].values, random_state=42, shuffle=True)#
    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())
    # combines rephrases to dataset.
    df_train = pd.concat([df_train, rephrase_df])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    print("Training labels distribution\n", df_train["sarcastic"].value_counts())
    print("Test labels distribution\n", df_test["sarcastic"].value_counts())

    # creates dataset to work with.
    DF_train = Dataset.TrainDataset(df_train, pretraine_path)
    DF_test = Dataset.TrainDataset(df_test, pretraine_path)

    DF_train_loader = DataLoader(dataset=DF_train, batch_size=batchsize, shuffle=True,
                                 num_workers=num_worker)
    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)
    return DF_train_loader, DF_test_loader

def loadTestData(batchsize=16, num_worker=2, pretraine_path="xlm-roberta-base"):
    # gets data and starts processing
    path = "data/task_A_En_test.csv"
    data = pd.read_csv(path, encoding='utf-8')
    data['text'] = data['text'].apply(lambda x:preprocess(x))

    # reports shape of it.
    print(data.shape)
    print(data.head())

    # starts to build dataset.
    DF_test = Dataset.TestDataset(data, pretraine_path)

    DF_test_loader = DataLoader(dataset=DF_test, batch_size=batchsize, shuffle=False,
                                num_workers=num_worker)

    # grabs sarcasm labels for fscore calculations.
    return DF_test_loader,data['sarcastic']
