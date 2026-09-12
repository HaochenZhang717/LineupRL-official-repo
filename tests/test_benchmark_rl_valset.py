import json

from ts_eval.benchmarks import rl_valset


def _write_fragments(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_val_messages(path, rows):
    with open(path, "w") as f:
        for image_abs, qa_list in rows:
            msg = [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "image", "image": image_abs}, {"type": "text", "text": "t"}]},
                {"role": "answer", "content": repr(qa_list)},
            ]
            f.write(json.dumps({"message": json.dumps(msg)}) + "\n")


def test_load_joins_series_from_fragments(tmp_path):
    frag_path = tmp_path / "fragments.jsonl"
    _write_fragments(frag_path, [
        {"id": 1, "image_path": "render/00000001.png", "series": [1.0, 2.0, 3.0], "len": 3, "ds_caption": "x"},
        {"id": 2, "image_path": "render/00000002.png", "series": [4.0, 5.0, 6.0], "len": 3, "ds_caption": "y"},
    ])
    val_path = tmp_path / "val_messages.jsonl"
    _write_val_messages(val_path, [
        (f"{tmp_path}/render/00000001.png", [("Q1 A) x B) y", "A"), ("Q2 A) x B) y", "B")]),
        (f"{tmp_path}/render/00000002.png", [("Q3 A) x B) y", "A")]),
    ])

    items = list(rl_valset.load(fragments_path=frag_path, val_messages_path=val_path))
    assert len(items) == 3
    assert items[0].series == [1.0, 2.0, 3.0]
    assert items[0].gold == "A"
    assert items[1].series == [1.0, 2.0, 3.0]
    assert items[1].gold == "B"
    assert items[2].series == [4.0, 5.0, 6.0]
    assert all(item.scoring_type == "letter_exact" for item in items)
    assert all(item.source_benchmark == "rl_valset" for item in items)


def test_load_respects_limit(tmp_path):
    frag_path = tmp_path / "fragments.jsonl"
    _write_fragments(frag_path, [{"id": 1, "image_path": "render/00000001.png", "series": [1.0], "len": 1, "ds_caption": "x"}])
    val_path = tmp_path / "val_messages.jsonl"
    _write_val_messages(val_path, [
        (f"{tmp_path}/render/00000001.png", [("Q1", "A"), ("Q2", "B"), ("Q3", "C")]),
    ])
    items = list(rl_valset.load(limit=2, fragments_path=frag_path, val_messages_path=val_path))
    assert len(items) == 2


def test_load_skips_images_missing_from_fragments(tmp_path):
    frag_path = tmp_path / "fragments.jsonl"
    _write_fragments(frag_path, [])
    val_path = tmp_path / "val_messages.jsonl"
    _write_val_messages(val_path, [(f"{tmp_path}/render/00000001.png", [("Q1", "A")])])
    items = list(rl_valset.load(fragments_path=frag_path, val_messages_path=val_path))
    assert items == []
