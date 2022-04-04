import pandas as pd

class CustomDataset():
    def __init__(self, phase):
        self.phase = phase
        def read_file(path):
            with open(path, "r", encoding="UTF_8") as f:
                dataset = f.read().splitlines()
            return dataset
        self.en = read_file(f'./wmt16/{self.phase}.en')
        self.de = read_file(f'./wmt16/{self.phase}.de')

        df_en = pd.DataFrame(self.en)
        df_de = pd.DataFrame(self.de)
        dataSet = pd.concat([df_en, df_de], axis=1)
        dataSet.columns = ['en', 'de']
        dataSet.to_csv(f'./wmt16/{self.phase}.csv', index=False)
        return


    def __getitem__(self, item):
        en = self.en[item]
        de = self.de[item]
        return en, de

    def __len__(self):
        return len(self.en)

