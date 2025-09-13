from gpusitter.gpu import query_gpu


def test_query_gpu() -> None:
    """Test querying GPU information."""
    gpus = query_gpu()
    assert gpus is not None
    assert isinstance(gpus, list)
    assert len(gpus) > 0
    assert isinstance(gpus[0], dict)
    assert "index" in gpus[0]
    assert "memory.total" in gpus[0]
    assert "memory.free" in gpus[0]
