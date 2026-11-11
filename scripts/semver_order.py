"""SemVer precedence and the release-pairing rule, shared by the Python scripts.

Mirrors src/extractor/mod.rs (`precedence`, `plan_pairs`) exactly, so a
predecessor rebuilt here is the one the extractor diffed against.
"""
import re

_CORE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def precedence(v):
    """Full SemVer 2.0.0 precedence key, or None if unparseable.

    (major, minor, patch), then 0 for a pre-release / 1 for a release, then the
    pre-release identifiers (numeric ones below alphanumeric ones, numeric
    compared as integers). Build metadata is ignored.
    """
    s = (v or "").strip().lstrip("v").split("+", 1)[0]
    core, _, pre = s.partition("-")
    m = _CORE.match(core)
    if not m:
        return None
    c = tuple(int(x) for x in m.groups())
    if not pre:
        return (c, 1, ())
    ids = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return (c, 0, ids)


def plan_prev(nums_in_publish_order):
    """{curr: prev} for every version that forms a pair (extractor plan_pairs).

    A version above the running maximum becomes the new maximum; it forms a
    pair with the old maximum only if its (major, minor, patch) is higher, so
    1.0.0 after 1.0.0-rc.1 forms no pair but 1.0.1 is then paired with 1.0.0.
    """
    prev_of, last, lk = {}, None, None
    for n in nums_in_publish_order:
        k = precedence(n)
        if k is None:
            continue
        if last is None:
            last, lk = n, k
        elif k > lk:
            if k[0] > lk[0]:
                prev_of[n] = last
            last, lk = n, k
    return prev_of


if __name__ == "__main__":  # same cases as the Rust tests
    assert plan_prev(["0.9.0", "1.0.0-rc.1", "1.0.0", "1.0.1"]) == {"1.0.0-rc.1": "0.9.0", "1.0.1": "1.0.0"}
    assert plan_prev(["1.0.0", "1.1.0-beta.1", "1.1.0-beta.2", "1.1.0", "1.2.0"]) == {"1.1.0-beta.1": "1.0.0", "1.2.0": "1.1.0"}
    assert plan_prev(["1.0.0-alpha.2", "1.0.0-alpha.10", "1.0.0", "2.0.0"]) == {"2.0.0": "1.0.0"}
    assert plan_prev(["1.0.0", "1.0.1", "0.9.5", "1.0.2"]) == {"1.0.1": "1.0.0", "1.0.2": "1.0.1"}
    assert plan_prev(["1.2.0", "1.1.5", "1.1.6", "1.3.0"]) == {"1.3.0": "1.2.0"}
    assert plan_prev(["1.0.0", "1.0.0+build", "1.0.0-rc.1", "1.0.1"]) == {"1.0.1": "1.0.0"}
    assert precedence("1.0.0-alpha") < precedence("1.0.0-alpha.1") < precedence("1.0.0-alpha.beta")
    assert precedence("1.0.0-1") < precedence("1.0.0-alpha") and precedence("1.0.0+x") == precedence("1.0.0")
    print("ok")
