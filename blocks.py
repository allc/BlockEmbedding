replaced_blocks = {
    'minecraft:cave_air': 'minecraft:air',
}

ignored_blocks = ['minecraft:fire']

keep_tags = {
    'minecraft:grass_block': ['snowy'],
    'minecraft:tall_grass': ['half'],
    'minecraft:lilac': ['half'],
    'minecraft:peony': ['half'],
    'minecraft:tall_seagrass': ['half'],
    'minecraft:large_fern': ['half'],
    'minecraft:spruce_log': ['half'],
    'minecraft:oak_log': ['axis'],
    'minecraft:birch_log': ['axis'],
    'minecraft:spruce_log': ['axis'],
    'minecraft:rail': ['shape'],
    'minecraft:oak_fence': ['east', 'waterlogged', 'south', 'north', 'west'],
    'minecraft:wall_torch': ['facing'],
    'minecraft:chest': ['waterlogged', 'facing', 'type'],
}

def normalise_block_state(block_state) -> None:
    # ignore if in ignored list
    if block_state.name in ignored_blocks:
        return
    # replace if in replace list
    if block_state.name in replaced_blocks:
        block_state.name = replaced_blocks[block_state.name]
    # keep only tags in list
    tags = {}
    if block_state.name in keep_tags:
        tags = {k: v for k, v in block_state.props.items() if k in keep_tags[block_state.name]}
    block_state.props = tags
