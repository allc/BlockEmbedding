from dataclasses import dataclass
import random
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import IterableDataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from PyAnvilEditor.pyanvil import World
from blocks import normalise_block_state
from utils import Word, load_block_index_map
from lightning.pytorch.loggers import WandbLogger

@dataclass
class BlockEmbeddingParams:
    # skipgram parameters
    CONTEXT_DISTANCE = 1  # window will be `(CONTEXT_DISTANCE * 2 + 1) ** 3`

    # model parameters
    VOCAB_SIZE = 70
    EMBED_DIM = 3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training parameters
    SUBSAMPLING_T = 85
    SUBSAMPLING_UNKNOWN_WORD_FREQUENCY = 4680568 # about four times the total token for building the vocab, the rate of sampling unknown words can be a bit small
    BATCH_SIZE = 8192
    N_EPOCHS = 5
    SAVE_EVERY_N_STEPS = 10000
    TOTAL_STEP_ESTIMATE = 150000

    # dataset parameters
    MINECRAFT_WORLD_DIR = 'saves/TrainingWorld3'
    # bigger number on each direction automatically being the end side, and is not included. The last a few x coordinates might be dropped if does not divide into workers
    MINECRAFT_WORLD_REGION = ((0, 0, 0), (-2000, 80, 64))


params = BlockEmbeddingParams()


class BlockEmbeddingModel(nn.Module):
    def __init__(self, params: BlockEmbeddingParams):
        super().__init__()
        self.t_embeddings = nn.Embedding(
            params.VOCAB_SIZE + 1,
            params.EMBED_DIM,
        )
        self.linear = nn.Linear(params.EMBED_DIM, params.VOCAB_SIZE + 1)

    def forward(self, inputs):
        target_embeddings = self.t_embeddings(inputs)
        output = self.linear(target_embeddings)
        return output


class LitBlockEmbeddingModel(L.LightningModule):
    def __init__(self, model, params: BlockEmbeddingParams):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.params = params

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x_hat = self.model(x)
        x_hat = x_hat.view(x_hat.size(0), -1)
        loss = F.cross_entropy(x_hat, y)
        self.log("train_loss", loss)

        sch = self.lr_schedulers()
        sch.step()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=params.TOTAL_STEP_ESTIMATE)
        return [optimizer], [scheduler]


class BlockDataset(IterableDataset):
    def __init__(self, params: BlockEmbeddingParams, block_index_map, world):
        self.block_index_map = block_index_map
        self.w = world
        self.vocab_size = params.VOCAB_SIZE
        self.offsets = []
        for dx in range(-params.CONTEXT_DISTANCE, params.CONTEXT_DISTANCE + 1):
            for dy in range(-params.CONTEXT_DISTANCE, params.CONTEXT_DISTANCE + 1):
                for dz in range(-params.CONTEXT_DISTANCE, params.CONTEXT_DISTANCE + 1):
                    self.offsets.append((dx, dy, dz))
        self.offset_i = 0
        ((xs, ys, zs), (xe, ye, ze)) = params.MINECRAFT_WORLD_REGION
        self.world_region = ((min(xs, xe), min(ys, ye), min(
            zs, ze)), (max(xs, xe), max(ys, ye), max(zs, ze)))
        self.target_block_i = 0
        self.subsampling_discard_p_list = self._get_subsampling_discard_p_list(params.SUBSAMPLING_T, params.SUBSAMPLING_UNKNOWN_WORD_FREQUENCY)

    def __iter__(self):
        return self

    def __next__(self):
        target_word, context_word = self._get_next_sample_words()
        # subsample context word
        context_word_index = context_word.index if context_word else self.vocab_size
        while random.random() < self.subsampling_discard_p_list[context_word_index]:
            target_word, context_word = self._get_next_sample_words()
            # subsample context word
            context_word_index = context_word.index if context_word else self.vocab_size
        target_word_index = target_word.index if target_word else self.vocab_size

        # advance indices
        self.offset_i = (self.offset_i + 1) % len(self.offsets)
        if self.offset_i == 0:
            self.target_block_i += 1
        if self.target_block_i % 1000 == 0 and self.offset_i < 1:
            print('Target block index:', self.target_block_i, self._get_target_block_coord(), 'context block offset:', self.offset_i)

        context_tensor = F.one_hot(torch.tensor(context_word_index), num_classes=self.vocab_size + 1)
        return (target_word_index, context_tensor.type(torch.float))
    
    def _get_subsampling_discard_p_list(self, T, unknown_word_frequency):
        n_tokens = sum([sum([w.frequency for w in l]) for l in self.block_index_map.values()])
        freq_list = [0] * (self.vocab_size + 1)
        for l in self.block_index_map.values():
            for w in l:
                freq_list[w.index] = w.frequency / n_tokens
        t = torch.quantile(torch.tensor(freq_list), torch.tensor(T / 100))
        freq_list[-1] = unknown_word_frequency / n_tokens
        discard_p = 1 - torch.sqrt(t / (torch.tensor(freq_list) + t))
        return discard_p
    
    def _get_next_sample_words(self):
        # get target and context block
        if not self._is_within_region(self._get_target_block_coord()):
            raise StopIteration
        target_block, context_block = self._get_sample_blocks()
        while not target_block or not context_block:
            if target_block is None:
                self.target_block_i += 1
                self.offset_i = 0
            elif not context_block:
                self.offset_i = (self.offset_i + 1) % len(self.offsets)
                if self.offset_i == 0:
                    self.target_block_i += 1
            if not self._is_within_region(self._get_target_block_coord()):
                raise StopIteration
            target_block, context_block = self._get_sample_blocks()

        # get target block index
        target_block_state = target_block.get_state().clone()
        normalise_block_state(target_block_state)
        target_word = self._get_word(
            target_block_state.name, target_block_state.props)
        # get context block index
        context_block_state = context_block.get_state().clone()
        normalise_block_state(context_block_state)
        context_word = self._get_word(
            context_block_state.name, context_block_state.props)
        return (target_word, context_word)

    def _get_word(self, name, tags) -> Word:
        words_with_name: list[Word] = self.block_index_map.get(name, [])
        word = None
        for w in words_with_name:
            if all([tags[k] == v for k, v in w.tags.items()]):
                word = w
                break
        return word

    def _get_sample_blocks(self):
        # get target block coord
        x, y, z = self._get_target_block_coord()
        # check if context block within region
        context_block_offset = self.offsets[self.offset_i]
        context_block_coord = (
            x + context_block_offset[0], y + context_block_offset[1], z + context_block_offset[2])
        if not self._is_within_region(context_block_coord):
            return (-1, None)
        # get target block
        try:
            target_block = self.w.get_block((x, y, z))
        except IndexError as err:
            print('Error getting target block at', (x, y, z), err)
            return (None, None)
        try:
            context_block = self.w.get_block(context_block_coord)
        except IndexError as err:
            print('Error getting context block at', context_block_coord, err)
            return (target_block, None)
        return (target_block, context_block)
    
    def _is_within_region(self, coord):
        return not any([coord[i] < self.world_region[0][i] or coord[i] >= self.world_region[1][i] for i in range(3)])
    
    def _get_target_block_coord(self):
        y = self.target_block_i % (
            self.world_region[1][1] - self.world_region[0][1]) + self.world_region[0][1]
        z = (self.target_block_i // (self.world_region[1][1] - self.world_region[0][1])) % (
            self.world_region[1][2] - self.world_region[0][2]) + self.world_region[0][2]
        x = self.target_block_i // ((self.world_region[1][1] - self.world_region[0][1]) * (
            self.world_region[1][2] - self.world_region[0][2])) + self.world_region[0][0]
        return (x, y, z)


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    dataset = worker_info.dataset
    r = dataset.world_region
    x_length = r[1][0] - r[0][0]
    x_start = r[0][0] + worker_id * (x_length // num_workers)
    x_end = x_start + (x_length // num_workers)
    dataset.world_region = (
        (x_start, r[0][1], r[0][2]), (x_end, r[1][1], r[1][2]))


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    # dataset
    block_index_map = load_block_index_map()
    world = World(params.MINECRAFT_WORLD_DIR, write=False)
    dataset = BlockDataset(params, block_index_map, world)
    dataloader = DataLoader(dataset, batch_size=params.BATCH_SIZE,
                            num_workers=8, worker_init_fn=worker_init_fn)

    # model
    model = BlockEmbeddingModel(params)
    lit_model = LitBlockEmbeddingModel(model, params)

    # train model
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=params.SAVE_EVERY_N_STEPS)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(log_model="all", project='BlockEmbedding')
    trainer = L.Trainer(callbacks=[checkpoint_callback, lr_monitor], logger=wandb_logger)
    trainer.fit(model=lit_model, train_dataloaders=dataloader)
