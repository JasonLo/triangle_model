import numpy as np
import pandas as pd
from PIL import Image
from typing import List

def gen_pkey(p_file="dataset/mappingv2.txt"):
    """Read phonological patterns from the mapping file
    See Harm & Seidenberg PDF file
    """

    mapping = pd.read_table(p_file, header=None, delim_whitespace=True)
    m_dict = mapping.set_index(0).T.to_dict("list")
    return m_dict



def get_pronunciation_fast(act, phon_key=None):
    if phon_key is None:
        phon_key = gen_pkey()
    phonemes = list(phon_key.keys())
    act10 = np.tile([v for k, v in phon_key.items()], 10)

    d = np.abs(act10 - act)
    d_mat = np.reshape(d, (38, 10, 25))
    sumd_mat = np.squeeze(np.sum(d_mat, 2))
    map_idx = np.argmin(sumd_mat, 0)
    out = str()
    for x in map_idx:
        out += phonemes[x]
    return out


def get_batch_pronunciations_fast(act, phon_key=None):
    if phon_key is None:
        phon_key = gen_pkey()
    return np.apply_along_axis(get_pronunciation_fast, 1, act, phon_key)


def stitch_fig(images:List[str], rows:int, columns:int) -> Image:
    """Stitch images in a grid"""
    assert len(images) <= (rows * columns)

    images = [Image.open(x) for x in images]

    # All images dimensions
    widths, heights = zip(*(im.size for im in images))

    # Max dims
    max_width = max(widths)
    max_height = max(heights)

    # Stitching
    stitched_image = Image.new('RGB', (max_width * columns, max_height * rows))

    x_offset = 0
    y_offset = 0

    for i, im in enumerate(images):
        stitched_image.paste(im, (x_offset, y_offset))
        if (i+1) % columns == 0:
            # New row every {columns} images
            y_offset += max_height
            x_offset = 0
        else:
            # New column
            x_offset += max_width

    return stitched_image
