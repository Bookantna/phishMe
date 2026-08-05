from __future__ import annotations

import math
import re
import unicodedata
from typing import NamedTuple

import numpy as np
from lxml import etree
from lxml import html as lxml_html
from scipy.sparse import csr_matrix

FEATURE_VERSION = "phishme-features-v1"
HASH_DIM = 1 << 18
NGRAM_RANGE = (3, 5)
COUNT_FEATURES = (
    "url_length",
    "domain_length",
    "tld_length",
    "subdomain_count",
    "letter_count",
    "digit_count",
    "obfuscated_count",
    "query_count",
    "equals_count",
    "ampersand_count",
    "special_count",
    "html_line_count",
    "largest_line_length",
    "iframe_count",
    "image_count",
    "css_count",
    "javascript_count",
    "self_ref_count",
    "empty_ref_count",
    "external_ref_count",
)
RATIO_FEATURES = ("letter_ratio", "digit_ratio", "obfuscated_ratio", "special_ratio")
BOOLEAN_FEATURES = (
    "is_domain_ip",
    "is_https",
    "has_title",
    "has_favicon",
    "is_responsive",
    "has_description",
    "external_form_submit",
    "has_social",
    "has_submit_button",
    "has_hidden_fields",
    "has_password_field",
    "mentions_bank",
    "mentions_pay",
    "mentions_crypto",
    "has_copyright",
)
NUMERIC_FEATURES = COUNT_FEATURES + RATIO_FEATURES + BOOLEAN_FEATURES
URL_NUMERIC_FEATURES = (
    "url_length",
    "domain_length",
    "tld_length",
    "subdomain_count",
    "letter_count",
    "digit_count",
    "obfuscated_count",
    "query_count",
    "equals_count",
    "ampersand_count",
    "special_count",
    "letter_ratio",
    "digit_ratio",
    "obfuscated_ratio",
    "special_ratio",
    "is_domain_ip",
    "is_https",
)
DOM_NUMERIC_FEATURES = tuple(name for name in NUMERIC_FEATURES if name not in URL_NUMERIC_FEATURES)

FROZEN_SINGLE_LABEL_SUFFIXES = frozenset(
    (
        "ai",
        "app",
        "au",
        "biz",
        "br",
        "cn",
        "co",
        "com",
        "dev",
        "edu",
        "gov",
        "info",
        "int",
        "io",
        "jp",
        "kr",
        "mil",
        "mx",
        "net",
        "nz",
        "org",
        "pl",
        "sa",
        "sg",
        "th",
        "tr",
        "uk",
        "us",
        "za",
    )
)
FROZEN_MULTI_LABEL_SUFFIXES = (
    "ac.th",
    "ac.uk",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.th",
    "co.uk",
    "co.za",
    "com.au",
    "com.br",
    "com.cn",
    "com.mx",
    "com.pl",
    "com.sa",
    "com.sg",
    "com.tr",
    "edu.au",
    "go.th",
    "gov.au",
    "gov.uk",
    "ltd.uk",
    "me.uk",
    "ne.jp",
    "net.au",
    "net.nz",
    "or.jp",
    "or.th",
    "org.nz",
    "org.uk",
)
_FROZEN_MULTI_LABEL_SUFFIX_LABELS = tuple(
    tuple(suffix.split("."))
    for suffix in sorted(FROZEN_MULTI_LABEL_SUFFIXES, key=lambda value: (-value.count("."), value))
)
_PERCENT_ESCAPE_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_IPV4_COMPONENT_RE = re.compile(r"^(0|[1-9][0-9]{0,2})$")
_IPV6_GROUP_RE = re.compile(r"^[0-9A-Fa-f]{1,4}$")
_HTML_PARSER = lxml_html.HTMLParser(recover=True, no_network=True)
_REFERENCE_ATTRIBUTES = (
    ("//a[@href]", "href"),
    ("//link[@href]", "href"),
    ("//script[@src]", "src"),
    ("//img[@src]", "src"),
    ("//iframe[@src]", "src"),
    ("//form[@action]", "action"),
)
_EMPTY_REF_SCHEMES = ("javascript:", "mailto:", "tel:", "data:", "about:")
_SOCIAL_DOMAINS = (
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "pinterest.com",
    "reddit.com",
    "t.co",
    "t.me",
    "telegram.me",
    "tiktok.com",
    "twitter.com",
    "wa.me",
    "whatsapp.com",
    "x.com",
    "youtube.com",
)
_BANK_TERMS = ("bank", "banking")
_PAY_TERMS = ("pay", "payment", "paypal", "checkout")
_CRYPTO_TERMS = ("crypto", "bitcoin", "btc", "ethereum", "wallet", "blockchain", "usdt")


class _FeatureUrlParts(NamedTuple):
    scheme: str
    hostname: str


def fnv1a_32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def iter_ngrams(namespace: str, text: str):
    normalized = unicodedata.normalize("NFKC", _as_text(text)).lower()
    prefix = f"{str(namespace).lower()}:"
    for size in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        for start in range(max(0, len(normalized) - size + 1)):
            yield f"{prefix}{normalized[start:start + size]}"


def url_numeric_features(url: str) -> dict[str, float]:
    raw = _as_text(url)
    parts = _split_url(raw)
    host = (parts.hostname or "").lower()
    url_length = len(raw)
    letter_count = sum(1 for character in raw if _is_unicode_letter(character))
    digit_count = sum(1 for character in raw if _is_unicode_number(character))
    obfuscated_count = len(_PERCENT_ESCAPE_RE.findall(raw))
    special_count = sum(1 for character in raw if not _is_unicode_alphanumeric(character))
    denominator = float(url_length or 1)
    suffix, subdomain_count = _suffix_and_subdomain_count(host)

    return {
        "url_length": float(url_length),
        "domain_length": float(len(host)),
        "tld_length": float(len(suffix)),
        "subdomain_count": float(subdomain_count),
        "letter_count": float(letter_count),
        "digit_count": float(digit_count),
        "obfuscated_count": float(obfuscated_count),
        "query_count": float(raw.count("?")),
        "equals_count": float(raw.count("=")),
        "ampersand_count": float(raw.count("&")),
        "special_count": float(special_count),
        "letter_ratio": letter_count / denominator,
        "digit_ratio": digit_count / denominator,
        "obfuscated_ratio": obfuscated_count / denominator,
        "special_ratio": special_count / denominator,
        "is_domain_ip": float(_is_ip_address(host)),
        "is_https": float(parts.scheme.lower() == "https"),
    }


def html_features(url: str, html: str) -> tuple[str, dict[str, float]]:
    title, values, _status = _html_features_with_status(url, html)
    return title, values


def vectorize(frame, include_dom=True, hash_dim=HASH_DIM):
    if hash_dim <= 0:
        raise ValueError("hash_dim must be positive")

    rows, columns, values = [], [], []
    records = frame.to_dict("records")
    for row_number, row in enumerate(records):
        raw_url = _as_text(row.get("url", ""))
        hashed = {fnv1a_32(token) % hash_dim for token in iter_ngrams("url", raw_url)}
        hashed.update(fnv1a_32(token) % hash_dim for token in iter_ngrams("title", row.get("title", "")))
        for column in sorted(hashed):
            rows.append(row_number)
            columns.append(column)
            values.append(1.0)
        if include_dom:
            url_values = url_numeric_features(raw_url)
            for offset, name in enumerate(NUMERIC_FEATURES):
                if name in URL_NUMERIC_FEATURES:
                    raw_value = url_values[name]
                else:
                    raw_value = row.get("dom_" + name, 0.0)
                value = _transform(name, raw_value)
                if value:
                    rows.append(row_number)
                    columns.append(hash_dim + offset)
                    values.append(value)

    width = hash_dim + (len(NUMERIC_FEATURES) if include_dom else 0)
    if not values:
        return csr_matrix((len(records), width), dtype=np.float32)
    return csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, columns)),
        shape=(len(records), width),
    )


def _html_features_with_status(url: str, html: str) -> tuple[str, dict[str, float], str]:
    text = _as_text(html)
    values = _zero_features_with_url(url)
    if not text.strip():
        return "", values, "empty"
    if "\x00" in text:
        return "", values, "parse_error"

    try:
        document = lxml_html.fromstring(text, parser=_HTML_PARSER)
    except (etree.ParserError, TypeError, ValueError):
        return "", values, "parse_error"

    title = _title(document)
    body_text = _visible_text(document)
    reference_counts = _reference_counts(url, document)

    values.update(
        {
            "html_line_count": float(len(text.splitlines())),
            "largest_line_length": float(max((len(line) for line in text.splitlines()), default=0)),
            "iframe_count": float(len(document.xpath("//iframe"))),
            "image_count": float(len(document.xpath("//img"))),
            "css_count": float(_css_count(document)),
            "javascript_count": float(len(document.xpath("//script"))),
            "self_ref_count": float(reference_counts["self"]),
            "empty_ref_count": float(reference_counts["empty"]),
            "external_ref_count": float(reference_counts["external"]),
            "has_title": float(bool(title)),
            "has_favicon": float(_has_favicon(document)),
            "is_responsive": float(bool(document.xpath("//meta[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'viewport']"))),
            "has_description": float(_has_description(document)),
            "external_form_submit": float(_has_external_form_submit(url, document)),
            "has_social": float(_has_social(document)),
            "has_submit_button": float(_has_submit_button(document)),
            "has_hidden_fields": float(bool(document.xpath("//input[translate(@type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'hidden']"))),
            "has_password_field": float(bool(document.xpath("//input[translate(@type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'password']"))),
            "mentions_bank": float(_contains_term(body_text, _BANK_TERMS)),
            "mentions_pay": float(_contains_term(body_text, _PAY_TERMS)),
            "mentions_crypto": float(_contains_term(body_text, _CRYPTO_TERMS)),
            "has_copyright": float("copyright" in body_text or "©" in body_text or "(c)" in body_text),
        }
    )
    return title, {name: values[name] for name in NUMERIC_FEATURES}, "ok"


def _zero_features_with_url(url: str) -> dict[str, float]:
    values = {name: 0.0 for name in NUMERIC_FEATURES}
    values.update(url_numeric_features(url))
    return values


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        if math.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _split_url(url: str):
    text = _as_text(url)
    scheme = ""
    rest = text
    scheme_index = text.find("://")
    if scheme_index >= 0:
        scheme = text[:scheme_index].lower()
        rest = text[scheme_index + 3 :]
    elif text.startswith("//"):
        rest = text[2:]

    authority_end = len(rest)
    for marker in ("/", "?", "#"):
        marker_index = rest.find(marker)
        if marker_index >= 0:
            authority_end = min(authority_end, marker_index)
    authority = rest[:authority_end]

    at_index = authority.rfind("@")
    if at_index >= 0:
        authority = authority[at_index + 1 :]

    hostname = authority
    if hostname.startswith("["):
        bracket_index = hostname.find("]")
        hostname = hostname[1:bracket_index] if bracket_index >= 0 else hostname[1:]
    else:
        colon_index = hostname.rfind(":")
        if colon_index >= 0:
            hostname = hostname[:colon_index]
    return _FeatureUrlParts(scheme=scheme, hostname=hostname)


def _is_ip_address(host: str) -> bool:
    text = _as_text(host)
    return _is_ipv4_address(text) or _is_ipv6_address(text)


def _is_ipv4_address(host: str) -> bool:
    pieces = host.split(".")
    if len(pieces) != 4:
        return False
    for piece in pieces:
        if not _IPV4_COMPONENT_RE.fullmatch(piece) or int(piece) > 255:
            return False
    return True


def _is_ipv6_address(host: str) -> bool:
    if ":" not in host:
        return False
    if any(character not in "0123456789abcdefABCDEF:." for character in host):
        return False

    text = host
    if "." in text:
        last_colon = text.rfind(":")
        if last_colon < 0 or not _is_ipv4_address(text[last_colon + 1 :]):
            return False
        text = f"{text[:last_colon]}:0:0"

    if text.count("::") > 1:
        return False
    if "::" in text:
        left, right = text.split("::")
        left_groups = left.split(":") if left else []
        right_groups = right.split(":") if right else []
        if not _are_ipv6_groups(left_groups) or not _are_ipv6_groups(right_groups):
            return False
        return len(left_groups) + len(right_groups) < 8

    groups = text.split(":")
    return _are_ipv6_groups(groups) and len(groups) == 8


def _are_ipv6_groups(groups: list[str]) -> bool:
    return all(_IPV6_GROUP_RE.fullmatch(group) for group in groups)


def _suffix_and_subdomain_count(host: str) -> tuple[str, int]:
    if not host or _is_ip_address(host):
        return "", 0
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 1:
        return "", 0

    multi_label_suffix = _matching_frozen_multi_label_suffix(labels)
    if multi_label_suffix:
        suffix_label_count = len(multi_label_suffix.split("."))
        return multi_label_suffix, max(0, len(labels) - suffix_label_count - 1)

    suffix = labels[-1]
    subdomain_count = (
        max(0, len(labels) - 2)
        if suffix in FROZEN_SINGLE_LABEL_SUFFIXES
        else max(0, len(labels) - 1)
    )
    return suffix, subdomain_count


def _registrable_domain_from_url(url: str) -> str:
    host = (_split_url(url).hostname or "").lower()
    if not host:
        return ""
    if _is_ip_address(host):
        return host
    suffix, subdomain_count = _suffix_and_subdomain_count(host)
    labels = [label for label in host.split(".") if label]
    if not suffix:
        return host
    suffix_label_count = len(suffix.split("."))
    domain_start = max(0, len(labels) - suffix_label_count - 1)
    if subdomain_count >= len(labels) - 1:
        return host
    return ".".join(labels[domain_start:])


def _matching_frozen_multi_label_suffix(labels: list[str]) -> str:
    for suffix_labels in _FROZEN_MULTI_LABEL_SUFFIX_LABELS:
        if len(labels) > len(suffix_labels) and tuple(labels[-len(suffix_labels) :]) == suffix_labels:
            return ".".join(suffix_labels)
    return ""


def _is_unicode_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _is_unicode_number(character: str) -> bool:
    return unicodedata.category(character).startswith("N")


def _is_unicode_alphanumeric(character: str) -> bool:
    category = unicodedata.category(character)
    return category.startswith(("L", "N"))


def _transform(name: str, value: float) -> float:
    value = _coerce_float(value)
    if name in COUNT_FEATURES:
        return math.log1p(max(0.0, value))
    if name in RATIO_FEATURES:
        return min(1.0, max(0.0, value))
    return 1.0 if value else 0.0


def _coerce_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _title(document) -> str:
    title = document.xpath("string((//title)[1])")
    return " ".join(title.split())


def _visible_text(document) -> str:
    text = " ".join(part.strip() for part in document.xpath("//body//text()") if part.strip())
    if not text:
        text = " ".join(part.strip() for part in document.xpath("//text()") if part.strip())
    return unicodedata.normalize("NFKC", text).lower()


def _css_count(document) -> int:
    stylesheets = document.xpath(
        "//link[contains(concat(' ', normalize-space(translate(@rel, "
        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')), ' '), ' stylesheet ')]"
    )
    return len(stylesheets) + len(document.xpath("//style"))


def _has_favicon(document) -> bool:
    links = document.xpath("//link[@rel or @href]")
    for link in links:
        rel = _attribute(link, "rel").lower()
        href = _attribute(link, "href").lower()
        if "icon" in rel.split() or "shortcut icon" in rel or href.endswith("favicon.ico"):
            return True
    return False


def _has_description(document) -> bool:
    metas = document.xpath("//meta[@name or @property]")
    for meta in metas:
        name = _attribute(meta, "name").lower()
        prop = _attribute(meta, "property").lower()
        content = _attribute(meta, "content").strip()
        if content and (name == "description" or prop in {"description", "og:description"}):
            return True
    return False


def _has_external_form_submit(url: str, document) -> bool:
    for form in document.xpath("//form[@action]"):
        if _classify_reference(url, _attribute(form, "action")) == "external":
            return True
    return False


def _has_social(document) -> bool:
    for node in document.xpath("//*[@href]"):
        href = _attribute(node, "href").lower()
        if any(domain in href for domain in _SOCIAL_DOMAINS):
            return True
    return False


def _has_submit_button(document) -> bool:
    submit_inputs = document.xpath(
        "//input[translate(@type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'submit']"
    )
    submit_buttons = document.xpath(
        "//button[not(@type) or translate(@type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz') = 'submit']"
    )
    return bool(submit_inputs or submit_buttons)


def _reference_counts(url: str, document) -> dict[str, int]:
    counts = {"self": 0, "empty": 0, "external": 0}
    for xpath, attribute in _REFERENCE_ATTRIBUTES:
        for node in document.xpath(xpath):
            counts[_classify_reference(url, _attribute(node, attribute))] += 1
    return counts


def _classify_reference(base_url: str, reference: str) -> str:
    value = reference.strip()
    lower = value.lower()
    if not value or value.startswith("#") or lower.startswith(_EMPTY_REF_SCHEMES):
        return "empty"
    target = _resolve_reference_url(base_url, value)
    target_domain = _registrable_domain_from_url(target)
    base_domain = _registrable_domain_from_url(base_url)
    if target_domain and base_domain and target_domain == base_domain:
        return "self"
    return "external"


def _resolve_reference_url(base_url: str, reference: str) -> str:
    if "://" in reference or reference.startswith("//"):
        return reference
    return base_url


def _attribute(node, name: str) -> str:
    value = node.get(name)
    return "" if value is None else str(value)


def _contains_term(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)
