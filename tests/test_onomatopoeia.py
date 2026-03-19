"""Tests for onomatopoeia detection."""
from babelvox.text.onomatopoeia import ONOMATOPOEIA_DB, process_onomatopoeia


class TestProcessOnomatopoeia:
    def test_single_detection(self):
        text, annotations = process_onomatopoeia("The door went boom")
        assert text == "The door went boom"  # text unchanged
        assert len(annotations) == 1
        assert annotations[0].type == "onomatopoeia"
        assert annotations[0].params["word"] == "boom"
        assert annotations[0].params["category"] == "impact"
        assert annotations[0].params["emphasis"] == "strong"

    def test_multiple_detections(self):
        text, annotations = process_onomatopoeia("crash and bang and sizzle")
        assert len(annotations) == 3

    def test_case_insensitive(self):
        _, annotations = process_onomatopoeia("BOOM goes the cannon")
        assert len(annotations) == 1
        assert annotations[0].params["word"] == "boom"

    def test_no_onomatopoeia(self):
        text, annotations = process_onomatopoeia("The cat sat on the mat")
        assert text == "The cat sat on the mat"
        assert annotations == []

    def test_word_boundary(self):
        # "boom" inside "boomerang" should not match
        _, annotations = process_onomatopoeia("A boomerang flew past")
        assert len(annotations) == 0

    def test_annotation_positions(self):
        text = "A crash happened"
        _, annotations = process_onomatopoeia(text)
        assert annotations[0].start == 2
        assert annotations[0].end == 7
        assert text[annotations[0].start:annotations[0].end] == "crash"

    def test_empty_string(self):
        text, annotations = process_onomatopoeia("")
        assert text == ""
        assert annotations == []


class TestOnomatopoeiaCoverage:
    def test_db_has_entries(self):
        assert len(ONOMATOPOEIA_DB) >= 50

    def test_all_entries_have_required_fields(self):
        for word, meta in ONOMATOPOEIA_DB.items():
            assert "category" in meta, f"{word} missing category"
            assert "emphasis" in meta, f"{word} missing emphasis"
            assert meta["emphasis"] in ("strong", "medium", "weak"), \
                f"{word} has invalid emphasis: {meta['emphasis']}"
