import pandas as pd

from phishme.data import split_local


def test_split_has_no_sample_or_domain_overlap():
    rows = []
    for domain_id in range(25):
        for page_id in range(2):
            rows.append(
                {
                    "url": f"https://d{domain_id}.example/p{page_id}",
                    "title": "login",
                    "label": domain_id % 2,
                    "group": f"d{domain_id}.example",
                    "sample_id": f"{domain_id}-{page_id}",
                }
            )
    parts = split_local(pd.DataFrame(rows))
    assert set(parts) == {"train", "validation", "test"}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert set(parts[left].group).isdisjoint(parts[right].group)
        assert set(parts[left].sample_id).isdisjoint(parts[right].sample_id)
    assert all(set(part.label) == {0, 1} for part in parts.values())
