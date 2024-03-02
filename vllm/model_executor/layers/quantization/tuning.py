# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
import os
import pickle
from typing import Dict, Tuple
import hidet

from .perf_model import Config


class Cache:
    def get(self, *args):
        raise NotImplementedError()
    
    def put(self, *args):
        raise NotImplementedError()

    def contains(self, *args):
        raise NotImplementedError()


class FileCache(Cache):
    def __init__(self, cache_file):
        self.cache_file = cache_file
    
    def load(self):
        raise NotImplementedError()

    def save(self, cache):
        raise NotImplementedError()


class LinearCache(FileCache):
    def __init__(self, cache_file):
        super().__init__(cache_file)
        self.cache: Dict[Tuple[int, int, int], Config] = {}
        self.load()

    def load(self):
        if len(self.cache) > 0:
            pass
        elif not os.path.exists(self.cache_file):
            self.cache = {}
        else:
            with open(self.cache_file, 'rb') as f:
                num_cache_items = pickle.load(f) 
                for _ in range(num_cache_items):
                    key = pickle.load(f)
                    value = pickle.load(f)
                    self.cache[key] = value

    def save(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(len(self.cache), f)
            for key, value in self.cache.items():
                pickle.dump(key, f)
                pickle.dump(value, f)

    def get(self, m, n, k):
        return self.cache.get((m, n, k), None)
    
    def put(self, m, n, k, Config):
        self.cache[(m, n, k)] = Config

    def contains(self, m, n, k):
        return (m, n, k) in self.cache


def linear_cache(name: str):
    cache_file = hidet.utils.cache_file(name)
    return LinearCache(cache_file)