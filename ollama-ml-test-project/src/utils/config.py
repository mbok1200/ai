def load_config():
    import yaml

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config