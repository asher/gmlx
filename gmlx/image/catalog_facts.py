"""The measured facts of the image catalog, printed whole by
``scripts/refresh_pull_catalog.py`` and pasted here. Edit nothing by
hand; ``catalog.RECIPES`` holds the words."""

# Each recipe's DiT file size per quant, from its repo listing at main.
DIT_BYTES: dict = {
    'z-image-turbo': {
        'Q3_K_M': 4186161216,
        'Q4_K_M': 5017613376,
        'Q5_K_M': 5574444096,
        'Q6_K': 5910505536,
        'Q8_0': 7224707136,
        'BF16': 12311939136,
    },
    'qwen-image': {
        'Q3_K_M': 3168290528,
        'Q4_K_M': 4199565024,
        'Q5_K_M': 5390223072,
        'Q6_K': 6271551200,
        'Q8_0': 7640860384,
        'F16': 14230275808,
    },
    'qwen-image-turbo': {
        'AD-Q3_K': 3599812320,
        'AD-Q4_K': 4201694944,
        'AD-Q5_K': 5399168736,
        'AD-Q6_K': 6711985888,
        'Q8_0': 7591554784,
        'BF16': 14230275808,
    },
    'krea-2-turbo': {
        'Q3_K_M': 6014825808,
        'Q4_K_M': 7486289232,
        'Q5_K_M': 8871195984,
        'Q6_K': 10583638350,
        'Q8_0': 13705958734,
    },
    'moss-image': {
        'Q3_K_M': 2891084096,
        'Q4_K_M': 3702092096,
        'Q5_K_M': 4114968896,
        'Q6_K': 4553650496,
        'Q8_0': 5705576768,
    },
    'moss-image-base': {
        'Q3_K_M': 2891084096,
        'Q4_K_M': 3702092096,
        'Q5_K_M': 4114968896,
        'Q6_K': 4553650496,
        'Q8_0': 5705576768,
    },
    'flux-2-klein': {
        'Q4_K_M': 2604311104,
        'Q8_0': 4300644928,
    },
    'flux-2-klein-base': {
        'Q4_K_M': 2604311104,
        'Q8_0': 4300644928,
    },
    'flux-2-klein-9b': {
        'Q4_K_M': 5909829920,
        'Q8_0': 9978304800,
    },
    'flux-2-klein-base-9b': {
        'Q4_K_M': 5909829920,
        'Q8_0': 9978304800,
    },
    'flux-2-klein-9b-kv': {
        'Q4_K_M': 5716891936,
        'Q8_0': 9978304800,
    },
    'flux-3-klein': {
        'Q4_K_M': 3391951648,
        'Q8_0': 6072111904,
    },
    'flux-3-klein-base': {
        'Q4_K_M': 3371504288,
        'Q8_0': 6051664544,
    },
}
