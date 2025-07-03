import os
from helpers.helpers_fn import get_all_files

class Automatization:
    def __init__(self, path):
        self.path = path
        self.files = get_all_files(path)

    def run(self):
        # Тут можна додати логіку для обробки файлів
        for file in self.files:
            print(f"Processing file: {file}")