from collections import defaultdict
import json
from PyAnvilEditor.pyanvil import World
import numpy
from blocks import normalise_block_state

START = (0, 0, 0)
FINISH = (-2000, 80, 64) # not included

TARGET_NUM_BLOCKS = 1556480

volume = numpy.prod([abs(FINISH[i] - START[i]) for i in range(3)])
print('Total sampling area volume:', volume)

skipping = pow(volume / TARGET_NUM_BLOCKS, 1/2) # only skipping on x, z direction, not y
print('Skipping:', skipping)
step = int(skipping) + 1

vocab = defaultdict(lambda: 0)

with World('saves/TrainingWorld3', write=False) as w:
    for x in range(min(START[0], FINISH[0]), max(START[0], FINISH[0]), step):
        for y in range(min(START[1], FINISH[1]), max(START[1], FINISH[1])):
            for z in range(min(START[2], FINISH[2]), max(START[2], FINISH[2]), step):
                pos = (x, y, z)
                try:
                    block = w.get_block(pos)
                    block_state = block.get_state().clone()
                    normalise_block_state(block_state)
                    if block_state not in vocab:
                        print(block_state, len(vocab))
                    vocab[block_state] += 1
                except IndexError as e: # errors on some chunk sections of air blocks above river
                    print(pos, e)

with open('vocabs.txt', 'w') as f:
    vocab = sorted(vocab.items(), key=lambda kv: kv[1], reverse=True)
    for i, (block_state, frequency) in enumerate(vocab):
        name = block_state.name
        tags = block_state.props
        f.write(f'{i} {name} {json.dumps(tags)} {frequency}\n')