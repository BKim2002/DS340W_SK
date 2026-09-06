from scripts.verify_checkpoint_reload import _target_max_length


class FakeTokenizer:
    def encode(self, label: str) -> list[int]:
        return {"0": [3, 632, 1], "1": [209, 1]}[label]


def test_reload_uses_the_authors_longest_sst2_label_length() -> None:
    assert _target_max_length(FakeTokenizer()) == 3
