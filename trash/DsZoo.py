# dataset
from torch.utils.data import DataLoader

from DsPolyU  import load_data
from trash.DsCasiaM import load_data as load_data_cs
from trash.DsIITD import load_data as load_data_iitd
from trash.DsTJPPV import load_data as load_data_tjppv


def get_data(config,train_ratio = 1,sample_ratio = 0.666):
    if "PolyU" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data(training=True,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data(training=False,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        # R G B R-B N
        return train_loader, test_loader, num_train, num_test
    if "CasiaM" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data_cs(training=True,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data_cs(training=False,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        # R B B R-B N
        return train_loader, test_loader, num_train, num_test
    if "IITD" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data_iitd(training=True,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data_iitd(training=False,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        # R G B R-B N
        return train_loader, test_loader, num_train, num_test
    if "TJPPV" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data_tjppv(training=True,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data_tjppv(training=False,train_ratio = 1,sample_ratio = 0.666), batch_size=batch_size, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        # R G B R-B N
        return train_loader, test_loader, num_train, num_test