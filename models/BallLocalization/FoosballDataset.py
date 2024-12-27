import FoosballDataset

class FoosballDatasetLocalizer(FoosballDataset):

    def __init__(self, images_dir, json_path, transform=None, train=True):
        super().__init__(images_dir, json_path, transform, train)

    def __get__item(self, idx):
        pass

    