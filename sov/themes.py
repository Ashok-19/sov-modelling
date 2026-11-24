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

MERGE_SCHEMA = {
    "type": "object",
    "properties": {
        "groups": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": {"type": "integer"}},
                    "theme": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["ids", "theme", "description"],
            },
        }
    },
    "required": ["groups"],
}

SYSTEM = (
    "You are an analyst reviewing social media posts about multiple sclerosis (MS) and "
    "its treatments, such as Ocrevus. Posts are in English or Spanish. Always answer in English."
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
        "- theme: a short, specific name (3 to 8 words) for what these posts have in common, "
        "specific enough to tell this group apart from the contrast posts. Avoid generic names "
        "like 'MS discussion' or 'Patient experiences'.\n"
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


def merge_themes(client, topic, named, vecs, labels):
    """Ask the LLM which theme names describe the same thing. No target count.

    Returns {cluster label: (final theme, final description)}.
    """
    ids = sorted(named)
    if len(ids) < 2:
        return {c: (named[c]["theme"], named[c]["description"]) for c in ids}
    cents = centroids(vecs, labels)
    cmat = np.stack([cents[c] for c in ids])
    sims = cmat @ cmat.T

    lines = []
    for i, c in enumerate(ids):
        closest = [j for j in np.argsort(-sims[i]) if j != i][:3]
        hint = ", ".join(f"#{ids[j]} ({sims[i, j]:.2f})" for j in closest)
        lines.append(f"#{c}: {named[c]['theme']} ({named[c]['size']} posts)\n"
                     f"    {named[c]['description']}\n    most similar by embedding: {hint}")
    user = (
        f"Parent topic: {topic}\n\nThese themes were found by clustering posts in this topic:\n\n"
        + "\n".join(lines)
        + "\n\nDecide which themes are really the same theme under different names and should be "
        "merged. Only merge themes that a reader would see as the same thing; keep themes apart "
        "if they cover different aspects, even when related. There is no target number of themes: "
        "it's fine if nothing gets merged.\n"
        "Return every theme id exactly once, in groups. A theme that isn't merged is its own group. "
        "For each group give the final theme name (short and specific) and a one or two sentence "
        "description. For a single-theme group you can keep its current name."
    )
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    answer, _ = client.chat_json(messages, MERGE_SCHEMA, "theme_merge")

    out, seen = {}, set()
    for g in answer["groups"]:
        members = [i for i in g["ids"] if i in named and i not in seen]
        seen.update(members)
        for i in members:
            out[i] = (g["theme"], g["description"])
        if len(members) > 1:
            log.info("Merged %s -> %s", [named[i]["theme"] for i in members], g["theme"])
    for c in ids:  # anything the model forgot keeps its own name
        if c not in out:
            log.warning("Merge step left out cluster %d; keeping its own name", c)
            out[c] = (named[c]["theme"], named[c]["description"])
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
