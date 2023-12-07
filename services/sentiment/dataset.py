from torch.utils.data import Dataset
from tqdm import tqdm
from razdel import sentenize, tokenize
import pandas as pd


class VKPostsDataset(Dataset):

    def __init__(self, data, with_entities=False):
        self.result = {}
        idx = 0

        for n in tqdm(range(len(data))):
            if with_entities==False:

                self.result[idx] = {
                    'id': data.iloc[n]['id'],
                    'message': data.iloc[n]['text'],
                    'sentiment': None,
                    'message_number': n,
                    'emotion': None,
                    'toxicity': None,
                    
                }
                idx += 1


            else:
                entities = [entity['substring'] for entity in data.iloc[n]["entity_info"]]

                if data.iloc[n]["entity_info"] is not None:
                    for sent in list(sentenize(data.iloc[n]['text'])):
                        for word in list(tokenize(sent.text)):
                            if word.text in entities:
                                self.result[idx] = {
                                    'context': sent.text,
                                    'message': data.iloc[n]['text'],
                                    'entity': word.text,
                                    'message_number': n,
                                    'sentiment': None,
                                    'emotion': None,
                                    'toxicity': None
                                }
                                idx += 1


    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        return self.result[idx]['message']

    def make_dataframe(self):
        return pd.DataFrame(self.result).transpose()


