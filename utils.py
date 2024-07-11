from collections import defaultdict
from dataclasses import dataclass
import json

@dataclass
class Word:
    index: int
    name: str
    tags: dict
    frequency: int

def load_block_index_map():
    block_index_map = defaultdict(list)
    with open('vocab.txt') as f:
        for line in f:
            line = line.split(maxsplit=2)
            index = int(line[0])
            name = line[1]
            line = line[2].rsplit(maxsplit=1)
            tags = json.loads(line[0])
            frequency = int(line[1])
            word = Word(index, name, tags, frequency)
            block_index_map[name].append(word)
    return block_index_map
