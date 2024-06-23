import random
import pandas as pd
from datasets import Dataset, DatasetDict


# Constants
STS_CSV_PATH = 'datasets/STS-Gold.csv'


def load_sts(test_ratio=0.2, seed=48):
    df = pd.read_csv(STS_CSV_PATH)
    df = df.drop('id', axis=1)
    df['polarity'] = df['polarity'].apply(lambda p: int(int(p) == 4))
    df['tweet'] = df['tweet'].apply(str.strip)
    df = df.rename(columns={'polarity': 'label', 'tweet': 'text'})

    # Split train and test
    data = df.to_dict('records')
    random.seed(seed)
    random.shuffle(data)
    n_test = int(len(data) * test_ratio)
    data_train = data[:-n_test]
    data_test = data[-n_test:]

    dataset = DatasetDict(
        train=Dataset.from_list(data_train),
        test=Dataset.from_list(data_test),
    )
    return dataset


if __name__ == '__main__':
    print(load_sts())
