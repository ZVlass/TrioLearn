from typing import Dict
from apps.core.models import Course, Book, Video, LearnerProfile
import random

# Map short topic codes (from the UI) to likely aliases found in vector keys or text
FOCUS_ALIASES = {
    "nlp":   ["nlp", "natural_language", "natural language", "transformer", "bert", "gpt", "llm", "text"],
    "cv":    ["cv", "computer_vision", "computer vision", "image", "vision", "cnn", "object detection"],
    "cloud": ["cloud", "aws", "azure", "gcp", "kubernetes", "docker", "serverless"],
    # add more as you use them: "ml": ["machine_learning", "ml", "sklearn", "regression", "classification"]
}

def _has_focus_item(obj, code: str) -> bool:
    """
    True if item matches the focus topic via:
    1) topic_vector contains the code or any alias (by substring in vector keys) with non-zero weight; OR
    2) title/description contains the code or any alias (case-insensitive).
    """
    if not code:
        return False
    code_l = code.lower()

    # 1) Vector-key / alias match
    tv = getattr(obj, "topic_vector", None)
    if isinstance(tv, dict) and tv:
        # normalize keys
        keys_l = [(k, k.lower()) for k in tv.keys()]
        # direct key match
        for raw_k, low_k in keys_l:
            if low_k == code_l and (tv.get(raw_k) or 0.0) > 0:
                return True
        # alias substring match against keys
        for alias in [code_l] + FOCUS_ALIASES.get(code_l, []):
            alias_l = alias.lower()
            for raw_k, low_k in keys_l:
                if alias_l in low_k and (tv.get(raw_k) or 0.0) > 0:
                    return True

    # 2) Text keyword match
    text = f"{getattr(obj, 'title', '')} {getattr(obj, 'description', '')}".lower()
    terms = [code_l] + FOCUS_ALIASES.get(code_l, [])
    return any(term in text for term in terms)


def item_matches_user_tags(item_topics: dict, user_tags: list, top_n: int = 3) -> bool:
    if not item_topics or not user_tags:
        return False
    item_top_tags = sorted(item_topics.items(), key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in item_top_tags[:top_n]]
    return any(tag in top_keys for tag in user_tags)

def update_modality_signal(profile: LearnerProfile, modality: str, strength: float = 1.0, decay: float = 0.9):
    if profile.course_prop is None: profile.course_prop = 0.0
    if profile.reading_prop is None: profile.reading_prop = 0.0
    if profile.video_prop is None: profile.video_prop = 0.0
    for key in ("course_prop", "reading_prop", "video_prop"):
        val = getattr(profile, key) or 0.0
        setattr(profile, key, decay * val)
    if modality == "course":
        profile.course_prop += (1 - decay) * strength
    elif modality == "reading":
        profile.reading_prop += (1 - decay) * strength
    elif modality == "video":
        profile.video_prop += (1 - decay) * strength
    profile.save(update_fields=["course_prop", "reading_prop", "video_prop", "last_active"])

def _quality_key_course(c: Course):
    return ((c.ratings or 0.0), (c.students_enrolled or 0), (c.popularity or 0.0))

def _quality_key_book(b: Book):
    return ((b.average_rating or 0.0), (b.ratings_count or 0))

def _quality_key_video(v: Video):
    return (0.0,)

def _matches_topics(profile: LearnerProfile, topic_vec: dict | None, focus_topic: str | None = None) -> bool:
    """
    - If focus_topic is given: accept any non-zero weight for that topic
      or presence in top tags.
    - Otherwise: use profile.matches_item.
    """
    if not topic_vec:
        return False
    if focus_topic:
        if topic_vec.get(focus_topic, 0.0) > 0:
            return True
        return item_matches_user_tags(topic_vec, [focus_topic], top_n=5)
    return profile.matches_item(topic_vec)

def recommend_for_profile(profile: LearnerProfile, k: int = 12, explore_eps: float = 0.2, focus_topic: str | None = None):
    """
    Returns:
      {
        "courses": list[Course],
        "books":   list[Book],
        "videos":  list[Video],
        "explore": list[Any],
        "surprise": Any|None
      }
    """
    weights: Dict[str, float] = profile.blended_format_distribution()  # {video, course, reading}

    # Candidate pools
    courses = list(Course.objects.all()[:300])
    books   = list(Book.objects.all()[:300])
    videos  = list(Video.objects.all()[:300])

    # Build in-topic pools ONCE
    if focus_topic:
        in_courses = [c for c in courses if _has_focus_item(c, focus_topic)]
        in_books   = [b for b in books   if _has_focus_item(b, focus_topic)]
        in_videos  = [v for v in videos  if _has_focus_item(v, focus_topic)]
    else:
        in_courses = [c for c in courses if _matches_topics(profile, getattr(c, "topic_vector", None), None)]
        in_books   = [b for b in books   if _matches_topics(profile, getattr(b, "topic_vector", None), None)]
        in_videos  = [v for v in videos  if _matches_topics(profile, getattr(v, "topic_vector", None), None)]

    # Off-topic via IDs (consistent with the in_* we just built)
    in_course_ids = {c.id for c in in_courses}
    in_book_ids   = {b.id for b in in_books}
    in_video_ids  = {v.id for v in in_videos}
    off_courses = [c for c in courses if c.id not in in_course_ids]
    off_books   = [b for b in books   if b.id not in in_book_ids]
    off_videos  = [v for v in videos  if v.id not in in_video_ids]

    # Quality sorts
    in_courses.sort(key=_quality_key_course, reverse=True)
    in_books.sort(key=_quality_key_book, reverse=True)
    in_videos.sort(key=_quality_key_video, reverse=True)
    off_courses.sort(key=_quality_key_course, reverse=True)
    off_books.sort(key=_quality_key_book, reverse=True)
    off_videos.sort(key=_quality_key_video, reverse=True)

    # Allocation
    k_explore = max(1, int(k * explore_eps))
    k_main = max(0, k - k_explore - 1)  # keep 1 for surprise
    alloc_course = max(1, round(weights.get("course", 0.0) * k_main))
    alloc_read   = max(1, round(weights.get("reading", 0.0) * k_main))
    alloc_video  = max(1, round(weights.get("video", 0.0) * k_main))
    total = alloc_course + alloc_read + alloc_video
    if total > k_main:
        while total > k_main:
            if alloc_course > 0 and total > k_main:
                alloc_course -= 1; total -= 1
            if alloc_read > 0 and total > k_main:
                alloc_read -= 1; total -= 1
            if alloc_video > 0 and total > k_main:
                alloc_video -= 1; total -= 1

    # Initial picks
    picks_courses = in_courses[:alloc_course]
    picks_books   = in_books[:alloc_read]
    picks_videos  = in_videos[:alloc_video]

    # Fallbacks that ACTUALLY update picks_*
    if focus_topic:
        in_courses = [c for c in courses if _has_focus_item(c, focus_topic)]
        in_books   = [b for b in books   if _has_focus_item(b, focus_topic)]
        in_videos  = [v for v in videos  if _has_focus_item(v, focus_topic)]
    else:
        in_courses = [c for c in courses if _matches_topics(profile, getattr(c, "topic_vector", None), None)]
        in_books   = [b for b in books   if _matches_topics(profile, getattr(b, "topic_vector", None), None)]
        in_videos  = [v for v in videos  if _matches_topics(profile, getattr(v, "topic_vector", None), None)]


    # Exploration
    explore_pool = off_courses[:k_explore] + off_books[:k_explore] + off_videos[:k_explore]
    random.shuffle(explore_pool)
    explore_picks = explore_pool[:k_explore]

    # Surprise
    surprise_pool = (off_videos[:25] + off_courses[:25] + off_books[:25]) or (videos + courses + books)
    surprise_item = random.choice(surprise_pool) if surprise_pool else None

    # Global safety net
    if not (picks_courses or picks_books or picks_videos):
        picks_courses = courses[:4]
        picks_books   = books[:4]
        picks_videos  = videos[:4]

    return {
        "courses": picks_courses,
        "books": picks_books,
        "videos": picks_videos,
        "explore": explore_picks,
        "surprise": surprise_item,
    }

    