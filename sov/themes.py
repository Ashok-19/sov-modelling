"""Naming clusters, merging names that mean the same thing, and placing noise posts.

Each cluster is named in a single LLM call that sees every post in it, and only
posts from that cluster. We never split a cluster into chunks. If a cluster is
too big for the context window we send as many posts as fit (most central
first), log a warning, and mark the theme as truncated in the output so nobody
mistakes it for a full read.

If the LLM fails to give a usable answer twice, we don't stop the run: the
cluster gets a placeholder name built from its keywords, logged as an error.
"""

import logging

import numpy as np

from . import progress

log = logging.getLogger(__name__)

NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {"type": "string", "minLength": 3, "maxLength": 100},
        "description": {"type": "string", "minLength": 3, "maxLength": 500},
        "confidence": {"type": "number"},
    },
    "required": ["theme", "description", "confidence"],
}

SAME_SCHEMA = {
    "type": "object",
    # The short reason comes first so the model thinks before it answers; the
    # length cap stops it from writing an essay.
    "properties": {"reason": {"type": "string", "maxLength": 300},
                   "same_theme": {"type": "boolean"}},
    "required": ["reason", "same_theme"],
}

GROUP_NAME_SCHEMA = {
    "type": "object",
    "properties": {"theme": {"type": "string", "maxLength": 100},
                   "description": {"type": "string", "maxLength": 500}},
    "required": ["theme", "description"],
}

# Each theme is compared with up to this many of its nearest neighbours, but
# only those closer than the topic's median theme-to-theme similarity, so
# clearly unrelated themes aren't checked.
MERGE_NEIGHBOURS = 4
MERGE_EXAMPLES = 6
# Two groups of themes join when at least this share of the pairs between them
# are judged the same. Requiring every pair let one noisy verdict keep obvious
# duplicates apart; a clear majority still stops unrelated themes chaining.
MERGE_AGREEMENT = 2 / 3

SYSTEM = (
    "You are a social listening analyst preparing a share of voice report on online "
    "conversation about multiple sclerosis (MS) for a pharmaceutical brand team (Ocrevus). "
    "Read posts carefully and describe what people are actually saying. Don't judge whether "
    "claims are true; just name them accurately. Answer in English."
)


def centroids(vecs, labels):
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        m = vecs[labels == c].mean(axis=0)
        out[c] = m / np.linalg.norm(m)
    return out


def _unique(posts):
    """Collapse exact repeats, keeping first-seen order, as (text, count) pairs.

    Identical retweets and shared headlines would otherwise fill the prompt
    and pull the name towards whatever got copied most. The model still sees
    how often each one appeared.
    """
    counts = {}
    for p in posts:
        counts[p] = counts.get(p, 0) + 1
    return list(counts.items())


def _fmt(posts):
    lines = []
    for i, (p, n) in enumerate(_unique(posts)):
        times = f" (posted {n} times)" if n > 1 else ""
        lines.append(f"[{i + 1}]{times} {p.strip()}")
    return "\n".join(lines)


def _name_messages(topic, keywords, posts):
    n_unique = len(_unique(posts))
    user = (
        f"Parent topic: {topic}\n\n"
        f"A clustering step grouped these {len(posts)} posts ({n_unique} distinct texts) "
        f"together. Keywords that stand out in the group: {', '.join(keywords)}\n\n"
        f"=== POSTS IN THIS GROUP ===\n{_fmt(posts)}\n\n"
        "Read every post. Work out the single thing most of them share. Then give:\n"
        "- theme: a short, specific name (3 to 8 words) that would make sense on its own as a "
        "row in a share of voice report. Name the actual claim, product, person, event or "
        "concern, for example 'Ivermectin promoted as an MS cure' rather than 'Alternative "
        "treatments', or 'Christina Applegate MS misdiagnosis news' rather than 'Celebrity "
        "news'. Don't use generic names like 'MS discussion', 'Patient experiences' or "
        "'Misinformation about MS', and don't just repeat the parent topic.\n"
        "- description: one or two sentences on what the posts say. If a sizeable minority "
        "talks about something else, mention it here.\n"
        "- confidence: from 0 to 1, how clearly the posts share one theme."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def name_clusters(client, topic, texts, vecs, labels, keywords):
    """Name every cluster in a topic. Returns {label: info dict}."""
    cents = centroids(vecs, labels)
    ids = list(cents)
    results = {}
    progress.ui.task(f"Naming clusters: {topic}", len(ids))
    for c in ids:
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(-(vecs[idx] @ cents[c]))]  # most central first
        posts = [texts[i] for i in order]

        messages = _name_messages(topic, keywords.get(c, []), posts)
        n_tokens = client.count_tokens(messages)
        truncated = False
        if n_tokens > client.prompt_budget():
            truncated = True
            keep = len(posts)
            while n_tokens > client.prompt_budget() and keep > 1:
                keep = int(keep * client.prompt_budget() / n_tokens * 0.97)
                messages = _name_messages(topic, keywords.get(c, []), posts[:keep])
                n_tokens = client.count_tokens(messages)
            log.warning("CLUSTER TOO LARGE FOR CONTEXT: topic %s cluster %d has %d posts; "
                        "only the %d most central fit in %d tokens. Marked as truncated.",
                        topic, c, len(posts), keep, client.ctx)
        log.info("Naming cluster %d (%d posts, %d prompt tokens)", c, len(posts), n_tokens)
        answer = _name_one(client, messages, topic, c, keywords.get(c, []))
        # "Other" is kept for posts that fit no theme.
        if answer["theme"].strip().lower() == "other":
            answer["theme"] = f"Other ({', '.join(keywords.get(c, [])[:3])})"
        results[c] = {**answer, "size": len(posts), "prompt_tokens": n_tokens,
                      "truncated": truncated, "keywords": keywords.get(c, [])}
        log.info("  -> %s (confidence %.2f)", answer["theme"], answer["confidence"])
        progress.ui.advance()
    return results


def _name_one(client, messages, topic, c, keywords):
    """Every cluster must come out with a theme name, so try harder before giving up.

    Thinking twice, then once more with thinking off, and only then a name
    built from the cluster's keywords (logged as an error so it gets looked at).
    """
    try:
        return client.chat_json(messages, NAME_SCHEMA, "cluster_name")[0]
    except ValueError:
        log.warning("Naming cluster %d in %s failed with thinking on; trying without", c, topic)
    try:
        return client.chat_json_no_thinking(messages, NAME_SCHEMA, "cluster_name")[0]
    except ValueError:
        pass
    log.error("Could not name cluster %d in %s; using its keywords as the name", c, topic)
    words = ", ".join(keywords[:4]) or f"cluster {c}"
    return {"theme": f"Unnamed theme ({words})",
            "description": "The LLM gave no usable name for this cluster; check it by hand.",
            "confidence": 0.0}


def _examples(texts, vecs, labels, c, cents, n=MERGE_EXAMPLES):
    """The n most typical distinct posts of a cluster, in full."""
    idx = np.where(labels == c)[0]
    out = []
    for i in idx[np.argsort(-(vecs[idx] @ cents[c]))]:
        if texts[i] not in out:
            out.append(texts[i])
        if len(out) == n:
            break
    return out


def _same_theme(client, topic, named, texts, vecs, labels, cents, a, b):
    side = lambda c: (f"Theme: {named[c]['theme']}\nDescription: {named[c]['description']}\n"
                      f"Example posts:\n{_fmt(_examples(texts, vecs, labels, c, cents))}")
    user = (
        f"Parent topic: {topic}\n\n=== GROUP A ===\n{side(a)}\n\n=== GROUP B ===\n{side(b)}\n\n"
        "Should A and B be reported as one theme in a share of voice report? Judge the core "
        "claim or subject, the thing an analyst would count: who or what the posts are about "
        "and what they say about it.\n"
        "Say yes when the core claim is the same, even if the posts cite a different source, "
        "describe a different mechanism, use different wording or tone, or name a more or less "
        "specific version of the same thing. For example 'WHO says COVID vaccination causes MS' "
        "and 'mRNA shots cause MS' are the same theme: both claim COVID vaccines cause MS. Posts "
        "that debunk a claim belong with the claim.\n"
        "Say no when the core claim or subject differs, even if related. For example 'COVID "
        "vaccines cause MS' and 'parasites cause MS' are different themes, and so are 'COVID "
        "vaccines cause MS' and 'general distrust of health authorities'."
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    try:
        answer, _ = client.chat_json(messages, SAME_SCHEMA, "same_theme")
    except ValueError:
        log.error("Merge check failed for %s vs %s; keeping them apart",
                  named[a]["theme"], named[b]["theme"])
        return False
    log.info("  %s  vs  %s: %s (%s)", named[a]["theme"], named[b]["theme"],
             "same" if answer["same_theme"] else "different", answer["reason"][:120])
    return bool(answer["same_theme"])


def merge_themes(client, topic, named, texts, vecs, labels):
    """Merge themes that are really the same thing. There's no target count.

    Asking the model to regroup a whole list at once made it lump everything
    into two or three vague buckets. So instead themes are compared one pair at
    a time, with example posts from both.

    Groups are only joined when a clear majority of the pairs between them are
    judged the same. Simply chaining "A is like B" and "B is like C" let a
    whole topic collapse into one theme, because C and A can be about quite
    different things; requiring every single pair to agree went too far the
    other way. Missing pairs are asked about when needed.

    Returns {cluster label: (final theme, final description)}.
    """
    ids = sorted(named)
    if len(ids) < 2:
        return {c: (named[c]["theme"], named[c]["description"]) for c in ids}
    cents = centroids(vecs, labels)
    cmat = np.stack([cents[c] for c in ids])
    sims = cmat @ cmat.T
    sim = {(a, b): float(sims[i, j]) for i, a in enumerate(ids) for j, b in enumerate(ids)}

    off_diag = sims[~np.eye(len(ids), dtype=bool)]
    close_enough = float(np.median(off_diag))
    pairs = set()
    for i in range(len(ids)):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        for j in order[:MERGE_NEIGHBOURS]:
            # The nearest neighbour is always checked; further ones only if close.
            if j == order[0] or sims[i, j] >= close_enough:
                pairs.add(tuple(sorted((ids[i], ids[j]))))
    log.info("Checking %d theme pairs for merges", len(pairs))

    verdicts = {}

    def same(a, b):
        key = tuple(sorted((a, b)))
        if key not in verdicts:
            verdicts[key] = _same_theme(client, topic, named, texts, vecs, labels, cents, *key)
            progress.ui.advance()
        return verdicts[key]

    progress.ui.task(f"Checking theme pairs for merges: {topic}", len(pairs))
    for a, b in sorted(pairs, key=lambda p: -sim[p]):
        same(a, b)

    # Join groups, closest pairs first, when most of the cross pairs agree.
    group_of = {c: frozenset([c]) for c in ids}
    for (a, b), is_same in sorted(verdicts.items(), key=lambda kv: -sim[kv[0]]):
        ga, gb = group_of[a], group_of[b]
        if not is_same or ga == gb:
            continue
        votes = [same(x, y) for x in ga for y in gb]
        if sum(votes) / len(votes) >= MERGE_AGREEMENT:
            merged = ga | gb
            for c in merged:
                group_of[c] = merged
        else:
            log.info("  Not joining %s with %s: only %d of %d pairs between them match",
                     [named[c]["theme"] for c in ga], [named[c]["theme"] for c in gb],
                     sum(votes), len(votes))

    groups = {g: sorted(g) for g in set(group_of.values())}
    out = {}
    for members in groups.values():
        if len(members) == 1:
            c = members[0]
            out[c] = (named[c]["theme"], named[c]["description"])
            continue
        listing = "\n".join(f"- {named[c]['theme']}: {named[c]['description']}" for c in members)
        examples = sum((_examples(texts, vecs, labels, c, cents, 3) for c in members), [])
        user = (f"Parent topic: {topic}\n\nThese groups of posts were judged to be the same theme:\n"
                f"{listing}\n\nExample posts:\n{_fmt(examples)}\n\n"
                "Give one short theme name (3 to 8 words) and a one or two sentence description "
                "that covers all of them. Keep the most specific thing they share: the actual "
                "claim, product, person or event. Don't fall back to a vaguer umbrella name.")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        try:
            answer, _ = client.chat_json(messages, GROUP_NAME_SCHEMA, "group_name")
        except ValueError:
            log.error("Could not rename a merged group; using its first member's name")
            answer = {"theme": named[members[0]]["theme"],
                      "description": named[members[0]]["description"]}
        log.info("Merged %s -> %s", [named[c]["theme"] for c in members], answer["theme"])
        for c in members:
            out[c] = (answer["theme"], answer["description"])
    return _unique_names(out, named)


def _unique_names(out, named):
    """Make sure separate groups don't end up sharing a name.

    Later steps group posts by theme name, so two groups the model kept apart
    but happened to name the same would silently become one. The second one
    gets its most telling keyword added to tell them apart.
    """
    owner = {}
    for c in sorted(out):
        name, desc = out[c]
        group = frozenset(k for k, v in out.items() if v == out[c])
        if name in owner and owner[name] != group:
            extra = next((k for k in named[c].get("keywords", []) if k.lower() not in name.lower()), str(c))
            log.info("Two separate themes were both named %r; renaming one to add %r", name, extra)
            new = f"{name} ({extra})"
            for k in group:
                out[k] = (new, desc)
            name = new
        owner[name] = group
    return out


def assign_noise(vecs, labels, final):
    """Give each noise post the nearest final theme, or 'Other' below a floor.

    The floor comes from the data: the 5th percentile of how similar clustered
    posts are to their own theme's centre. A noise post less similar than that
    to every theme doesn't really belong to any of them.
    """
    theme_of = np.array([final[l][0] if l >= 0 else None for l in labels], dtype=object)
    names = sorted({v[0] for v in final.values()})
    if not names:
        return theme_of, None, np.zeros(len(labels), dtype=bool), np.full(len(labels), np.nan)
    cents = []
    for name in names:
        m = vecs[theme_of == name].mean(axis=0)
        cents.append(m / np.linalg.norm(m))
    cents = np.stack(cents)
    name_idx = {n: i for i, n in enumerate(names)}

    clustered = labels >= 0
    own = np.array([vecs[i] @ cents[name_idx[theme_of[i]]] for i in np.where(clustered)[0]])
    floor = float(np.percentile(own, 5))

    reassigned = np.zeros(len(labels), dtype=bool)
    for i in np.where(~clustered)[0]:
        s = cents @ vecs[i]
        j = int(np.argmax(s))
        if s[j] >= floor:
            theme_of[i] = names[j]
            reassigned[i] = True
        else:
            theme_of[i] = "Other"

    # How close each post sits to its theme's centre; used to pick examples.
    closeness = np.array([float(vecs[i] @ cents[name_idx[t]]) if t in name_idx else np.nan
                          for i, t in enumerate(theme_of)])
    return theme_of, floor, reassigned, closeness
