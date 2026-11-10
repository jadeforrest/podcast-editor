import io

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from podedit.splice import envelope, splice_array, splice_stream


@st.composite
def ranges(draw, total):
    cuts = sorted(draw(st.sets(st.integers(0, total), max_size=12)))
    pairs = list(zip(cuts[0::2], cuts[1::2]))
    return [(a, b) for a, b in pairs if b > a]


@settings(max_examples=150)
@given(st.integers(1, 3000), st.data(), st.integers(1, 700), st.integers(0, 40))
def test_stream_matches_reference(total, data, chunk, fade):
    """Property: streaming in any chunk size equals the whole-array reference, with exact length."""
    rng = np.random.default_rng(total)
    audio = rng.uniform(-1, 1, size=(total, 2)).astype(np.float32)
    rs = data.draw(ranges(total))
    out = io.BytesIO()
    n = splice_stream(io.BytesIO(audio.tobytes()), out, rs, total, 2, fade, chunk_frames=chunk)
    got = np.frombuffer(out.getvalue(), dtype=np.float32).reshape(-1, 2)
    assert n == len(got) == sum(b - a for a, b in rs)
    np.testing.assert_allclose(got, splice_array(audio, rs, fade), atol=1e-7)


def test_fades_only_at_splices():
    audio = np.ones((100, 1), dtype=np.float32)
    out = splice_array(audio, [(0, 40), (60, 100)], fade=10)
    assert out[0, 0] == 1.0 and out[-1, 0] == 1.0     # file start / end untouched
    assert out[39, 0] < 0.1 and out[40, 0] < 0.1     # faded around the splice
    assert out[20, 0] == 1.0


def test_short_source_is_padded_with_silence():
    out = io.BytesIO()
    src = io.BytesIO(np.ones((50, 2), dtype=np.float32).tobytes())
    n = splice_stream(src, out, [(0, 80)], 80, 2, 0, on_progress=lambda f: None)
    got = np.frombuffer(out.getvalue(), dtype=np.float32).reshape(-1, 2)
    assert n == 80 and got[:50].min() == 1.0 and got[50:].max() == 0.0


def test_clipping_and_envelope():
    out = io.BytesIO()
    splice_stream(io.BytesIO(np.full((10, 1), 3.0, np.float32).tobytes()), out, [(0, 10)], 10, 1, 0)
    assert np.frombuffer(out.getvalue(), dtype=np.float32).max() == 1.0
    assert envelope(np.arange(5), 0, 5, 0, True, True).tolist() == [1.0] * 5
