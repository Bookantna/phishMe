import math

import pandas as pd

from phishme.features import (
    HASH_DIM,
    NUMERIC_FEATURES,
    _html_features_with_status,
    fnv1a_32,
    html_features,
    iter_ngrams,
    url_numeric_features,
    vectorize,
)


def test_fnv1a_matches_published_vector():
    assert fnv1a_32("hello") == 0x4F9F2CAB


def test_ngrams_are_unicode_normalized_and_namespaced():
    grams = set(iter_ngrams("URL", "ＡbC"))
    assert "url:abc" in grams


def test_vectorization_is_binary_for_duplicate_ngrams():
    frame = pd.DataFrame([{"url": "https://aaaa.example", "title": "", "label": 1}])
    matrix = vectorize(frame, include_dom=False)
    assert matrix.shape == (1, HASH_DIM)
    assert matrix.data.max() == 1.0


def test_numeric_counts_use_log1p():
    frame = pd.DataFrame(
        [{"url": "https://x.example", "title": "", "label": 0, "dom_image_count": 9}]
    )
    matrix = vectorize(frame, include_dom=True)
    assert any(math.isclose(float(value), math.log1p(9), rel_tol=1e-6) for value in matrix.data)


def test_url_numeric_features_are_recomputed_from_raw_url():
    url = "http://192.0.2.1/login?token=%2Fabc&step=2"
    values = url_numeric_features(url)

    assert values["url_length"] == len(url)
    assert values["domain_length"] == len("192.0.2.1")
    assert values["is_domain_ip"] == 1.0
    assert values["is_https"] == 0.0
    assert values["obfuscated_count"] == 1.0
    assert values["query_count"] == 1.0
    assert values["equals_count"] == 2.0
    assert values["ampersand_count"] == 1.0
    assert values["digit_ratio"] == values["digit_count"] / values["url_length"]


def test_html_features_extract_browser_computable_dom_values():
    title, values = html_features(
        "https://shop.example.com/login",
        """
        <html>
          <head>
            <title>Account Portal</title>
            <link rel="icon" href="/favicon.ico">
            <link rel="stylesheet" href="/site.css">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="description" content="Secure bank payment">
            <script src="/app.js"></script>
          </head>
          <body>
            <img src="/logo.png">
            <iframe src="/frame.html"></iframe>
            <form action="https://evil.example/submit">
              <input type="hidden" name="token">
              <input type="password" name="password">
              <button type="submit">Pay</button>
            </form>
            <a href="/account">Account</a>
            <a href="#">Empty</a>
            <a href="https://twitter.com/example">Social</a>
            <p>Copyright bank pay bitcoin.</p>
          </body>
        </html>
        """,
    )

    assert title == "Account Portal"
    assert set(values) == set(NUMERIC_FEATURES)
    assert values["has_title"] == 1.0
    assert values["has_favicon"] == 1.0
    assert values["is_responsive"] == 1.0
    assert values["has_description"] == 1.0
    assert values["iframe_count"] == 1.0
    assert values["image_count"] == 1.0
    assert values["css_count"] == 1.0
    assert values["javascript_count"] == 1.0
    assert values["external_form_submit"] == 1.0
    assert values["has_social"] == 1.0
    assert values["has_submit_button"] == 1.0
    assert values["has_hidden_fields"] == 1.0
    assert values["has_password_field"] == 1.0
    assert values["mentions_bank"] == 1.0
    assert values["mentions_pay"] == 1.0
    assert values["mentions_crypto"] == 1.0
    assert values["has_copyright"] == 1.0
    assert values["self_ref_count"] == 6.0
    assert values["empty_ref_count"] == 1.0
    assert values["external_ref_count"] == 2.0


def test_empty_and_malformed_html_return_zero_dom_values():
    url = "https://example.com/login"
    url_values = url_numeric_features(url)

    empty_title, empty_values, empty_status = _html_features_with_status(url, "")
    malformed_title, malformed_values, malformed_status = _html_features_with_status(url, "\x00")

    assert empty_title == ""
    assert malformed_title == ""
    assert empty_status == "empty"
    assert malformed_status == "parse_error"
    for values in (empty_values, malformed_values):
        assert set(values) == set(NUMERIC_FEATURES)
        for name in NUMERIC_FEATURES:
            if name in url_values:
                assert values[name] == url_values[name]
            else:
                assert values[name] == 0.0
