from sofia_distilled.chat_data import prompt_bank


def test_topic_families_never_leak_between_partitions():
    rows = prompt_bank()
    assert len({r["prompt"] for r in rows}) == len(rows)
    groups = {
        split: {r["topic_id"] for r in rows if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert not groups["train"] & groups["validation"]
    assert not groups["train"] & groups["test"]
    assert not groups["test"] & groups["validation"]
