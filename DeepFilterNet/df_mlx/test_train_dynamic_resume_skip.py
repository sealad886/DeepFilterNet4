from df_mlx.train_dynamic import maybe_skip_resume_batches


def test_maybe_skip_resume_batches_skips_when_resuming_same_epoch():
    data = iter(range(8))
    out, did_skip = maybe_skip_resume_batches(
        data,
        resume_from="checkpoint.safetensors",
        epoch=3,
        start_epoch=3,
        resume_batch_idx=2,
    )
    assert did_skip is True
    assert list(out) == [2, 3, 4, 5, 6, 7]


def test_maybe_skip_resume_batches_does_not_skip_other_epochs():
    data = iter(range(5))
    out, did_skip = maybe_skip_resume_batches(
        data,
        resume_from="checkpoint.safetensors",
        epoch=4,
        start_epoch=3,
        resume_batch_idx=2,
    )
    assert did_skip is False
    assert list(out) == [0, 1, 2, 3, 4]
