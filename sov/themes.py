"""Naming clusters, merging names that mean the same thing, and placing noise posts.

Each cluster is named in a single LLM call that sees every post in it. We never
split a cluster into chunks. If a cluster is too big for the context window we
send as many posts as fit (most central first), log a warning, and mark the
theme as truncated in the output so nobody mistakes it for a full read.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

N_CONTRAST = 25

NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "theme": {"type": "string"},
        "description": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["theme", "description", "confidence"],
}

SAME_SCHEMA = {
    "type": "object",
    "properties": {"same_theme": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["reason", "same_theme"],
}

GROUP_NAME_SCHEMA = {
    "type": "object",
    "properties": {"theme": {"type": "string"}, "description": {"type": "string"}},
    "required": ["theme", "description"],
}

# Each theme is only compared with its few nearest neighbours by embedding.
MERGE_NEIGHBOURS = 2
MERGE_EXAMPLES = 6

SYSTEM = (
    "You are an analyst reviewing social media posts about multiple sclerosis (MS) and "
    "its treatments, such as Ocrevus. Answer in English."
)


def centroids(vecs, labels):
    out = {}
    for c in sorted(set(labels)):
        if c < 0:
            continue
        m = vecs[labels == c].mean(axis=0)
        out[c] = m / np.linalg.norm(m)
    return out


def _fmt(posts):
    return "\n".join(f"[{i + 1}] {p.strip()}" for i, p in enumerate(posts))


def _name_messages(topic, keywords, posts, contrast):
    user = (
        f"Parent topic: {topic}\n\n"
        f"All posts in this group were placed together by a clustering step. "
        f"Top keywords for the group: {', '.join(keywords)}\n\n"
        f"=== POSTS IN THIS GROUP ({len(posts)}) ===\n{_fmt(posts)}\n\n"
        f"=== CONTRAST: posts from neighbouring groups (NOT in this group) ===\n{_fmt(contrast)}\n\n"
        "Read every post in the group. Then give:\n"
        "- theme: a short, specific name (3 to 8 words) for what these posts have in common. "
        "Name the actual claim, product, person, event or concern (for example 'Ivermectin "
        "promoted as an MS cure' rather than 'Alternative treatments'), specific enough to tell "
        "this group apart from the contrast posts. Avoid generic names like 'MS discussion', "
        "'Patient experiences' or 'Misinformation about MS'.\n"
        "- description: one or two sentences on what the posts talk about.\n"
        "- confidence: from 0 to 1, how clearly the posts share one theme."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def name_clusters(client, topic, texts, vecs, labels, keywords):
    """Name every cluster in a topic. Returns {label: info dict}."""
    cents = centroids(vecs, labels)
    ids = list(cents)
    cmat = np.stack([cents[c] for c in ids])
    results = {}
    for c in ids:
        idx = np.where(labels == c)[0]
        order = idx[np.argsort(-(vecs[idx] @ cents[c]))]  # most central first
        posts = [texts[i] for i in order]

        # Contrast posts: the most typical posts of the nearest other clusters.
        near = [ids[j] for j in np.argsort(-(cmat @ cents[c])) if ids[j] != c]
        contrast = []
        if near:
            per = max(1, N_CONTRAST // min(len(near), 5))
            for o in near:
                oidx = np.where(labels == o)[0]
                oidx = oidx[np.argsort(-(vecs[oidx] @ cents[o]))][:per]
                contrast += [texts[i] for i in oidx]
                if len(contrast) >= N_CONTRAST:
                    break
            contrast = contrast[:N_CONTRAST]

        messages = _name_messages(topic, keywords.get(c, []), posts, contrast)
        n_tokens = client.count_tokens(messages)
        truncated = False
        if n_tokens > client.prompt_budget():
            truncated = True
            keep = len(posts)
            while n_tokens > client.prompt_budget() and keep > 1:
                keep = int(keep * client.prompt_budget() / n_tokens * 0.97)
                messages = _name_messages(topic, keywords.get(c, []), posts[:keep], contrast)
                n_tokens = client.count_tokens(messages)
            log.warning("CLUSTER TOO LARGE FOR CONTEXT: topic %s cluster %d has %d posts; "
                        "only the %d most central fit in %d tokens. Marked as truncated.",
                        topic, c, len(posts), keep, client.ctx)
        log.info("Naming cluster %d (%d posts, %d prompt tokens)", c, len(posts), n_tokens)
        answer, _ = client.chat_json(messages, NAME_SCHEMA, "cluster_name")
        results[c] = {**answer, "size": len(posts), "prompt_tokens": n_tokens,
                      "truncated": truncated, "keywords": keywords.get(c, [])}
        log.info("  -> %s (confidence %.2f)", answer["theme"], answer["confidence"])
    return results


def _examples(texts, vecs, labels, c, cents, n=MERGE_EXAMPLES):
    idx = np.where(labels == c)[0]
    idx = idx[np.argsort(-(vecs[idx] @ cents[c]))][:n]
    return [texts[i][:600] for i in idx]


def merge_themes(client, topic, named, texts, vecs, labels):
    """Merge themes that are really the same thing. There's no target count.

    Asking the model to regroup a whole list at once made it lump everything
    into two or three vague buckets. So instead each theme is checked against
    its nearest neighbours one pair at a time, with example posts from both,
    and merged only when the model says they make the same specific point.
    Merged groups then get one fresh name.

    Returns {cluster label: (final theme, final description)}.
    """
    ids = sorted(named)
    if len(ids) < 2:
        return {c: (named[c]["theme"], named[c]["description"]) for c in ids}
    cents = centroids(vecs, labels)
    cmat = np.stack([cents[c] for c in ids])
    sims = cmat @ cmat.T

    pairs = set()
    for i in range(len(ids)):
        for j in [j for j in np.argsort(-sims[i]) if j != i][:MERGE_NEIGHBOURS]:
            pairs.add(tuple(sorted((ids[i], ids[j]))))

    parent = {c: c for c in ids}

    def root(c):
        while parent[c] != c:
            c = parent[c]
        return c

    for a, b in sorted(pairs):
        side = lambda c: (f"Theme: {named[c]['theme']}\nDescription: {named[c]['description']}\n"
                          f"Example posts:\n{_fmt(_examples(texts, vecs, labels, c, cents))}")
        user = (
            f"Parent topic: {topic}\n\n=== GROUP A ===\n{side(a)}\n\n=== GROUP B ===\n{side(b)}\n\n"
            "Are A and B the same theme? Say yes only if they make the same specific claim or "
            "discuss the same specific thing, so that reporting them under one name would lose "
            "nothing an analyst cares about. If they are related but about different claims, "
            "products, people or events, say no."
        )
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        answer, _ = client.chat_json(messages, SAME_SCHEMA, "same_theme")
        log.info("  %s  vs  %s: %s (%s)", named[a]["theme"], named[b]["theme"],
                 "same" if answer["same_theme"] else "different", answer["reason"][:120])
        if answer["same_theme"]:
            parent[root(b)] = root(a)

    groups = {}
    for c in ids:
        groups.setdefault(root(c), []).append(c)
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
                "Give one short, specific theme name (3 to 8 words) and a one or two sentence "
                "description that covers all of them. Keep it as specific as the names above.")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        answer, _ = client.chat_json(messages, GROUP_NAME_SCHEMA, "group_name")
        log.info("Merged %s -> %s", [named[c]["theme"] for c in members], answer["theme"])
        for c in members:
            out[c] = (answer["theme"], answer["description"])
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
        return theme_of, None, np.zeros(len(labels), dtype=bool)
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
    return theme_of, floor, reassigned
