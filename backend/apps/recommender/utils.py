# apps/recommendation/utils.py

def item_matches_user_tags(item_topics: dict, user_tags: list, top_n: int = 3) -> bool:
    if not item_topics or not user_tags:
        return False
    item_top_tags = sorted(item_topics.items(), key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in item_top_tags[:top_n]]
    return any(tag in top_keys for tag in user_tags)
