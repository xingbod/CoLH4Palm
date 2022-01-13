from DsPolyU  import load_data
from DsCasiaM  import load_data_cs



def get_data(config):
    if "PolyU" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data(training=True), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data(training=False), batch_size=64, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        return train_loader, test_loader, num_train, num_test
    if "CasiaM" in config["dataset"]:
        batch_size = config["batch_size"]
        train_loader = DataLoader(load_data_cs(training=True), batch_size=batch_size, shuffle=True, num_workers=8,       pin_memory=True,prefetch_factor=2)  # ,prefetch_factor=2
        test_loader = DataLoader(load_data_cs(training=False), batch_size=64, shuffle=False)  # ,prefetch_factor=2
        # dataset_loader = test_loader
        num_train = len(train_loader.dataset)
        num_test = len(test_loader.dataset)
        print(len(train_loader.dataset))
        print(len(test_loader.dataset))
        return train_loader, test_loader, num_train, num_test