from typing import Dict
from apps.core.models import Course, Book, Video, LearnerProfile
import random

def item_matches_user_tags(item_topics: dict, user_tags: list, top_n: int = 3) -> bool:
    if not item_topics or not user_tags:
        return False
    item_top_tags = sorted(item_topics.items(), key=lambda x: x[1], reverse=True)
    top_keys = [k for k, _ in item_top_tags[:top_n]]
    return any(tag in top_keys for tag in user_tags)


def update_modality_signal(profile: LearnerProfile, modality: str, strength: float = 1.0, decay: float = 0.9):
    """
    Exponential moving average update of modality props based on a click.
    decay ~0.9 keeps history; lower means faster adaptation.
    """
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
    # Fallback-friendly sort: higher is better
    return (
        (c.ratings or 0.0),
        (c.students_enrolled or 0),
        (c.popularity or 0.0),
    )

def _quality_key_book(b: Book):
    return (
        (b.average_rating or 0.0),
        (b.ratings_count or 0),
    )

def _quality_key_video(v: Video):
    # If you later compute watch stats, include them here
    return (0.0,)


def _matches_topics(profile: LearnerProfile, topic_vec: dict | None, focus_topic: str | None = None, thresh: float = 0.1) -> bool:
    if not topic_vec:
        return False
    if focus_topic:
        return topic_vec.get(focus_topic, 0.0) >= thresh
    return profile.matches_item(topic_vec)


def recommend_for_profile(profile: LearnerProfile, k: int = 12, explore_eps: float = 0.2, focus_topic: str | None = None):
    """
    Blend modalities per the profile's blended distribution and include exploration.
    Returns dict with top lists and a surprise item.
    """
    weights: Dict[str, float] = profile.blended_format_distribution()  # {video, course, reading}

    # Fetch candidates (oversample for filtering)
    courses = list(Course.objects.all()[:300])
    books   = list(Book.objects.all()[:300])
    videos  = list(Video.objects.all()[:300])

    # In-topic vs off-topic splits (respecting optional focus_topic)
    in_courses = [c for c in courses if _matches_topics(profile, c.topic_vector, focus_topic)]
    in_books   = [b for b in books   if _matches_topics(profile, b.topic_vector, focus_topic)]
    in_videos  = [v for v in videos  if _matches_topics(profile, v.topic_vector, focus_topic)]

    # Faster "off" sets using IDs (avoid O(n^2) list membership)
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

    # Allocation per modality
    k_explore = max(1, int(k * explore_eps))
    k_main = max(0, k - k_explore - 1)  # leave 1 slot for "surprise"

    alloc_course = max(1, round(weights.get("course", 0.0) * k_main))
    alloc_read   = max(1, round(weights.get("reading", 0.0) * k_main))
    alloc_video  = max(1, round(weights.get("video", 0.0) * k_main))

    # Normalize if over
    total = alloc_course + alloc_read + alloc_video
    if total > k_main:
        # trim in a loop until fits
        while total > k_main:
            if alloc_course > 0 and total > k_main: alloc_course -= 1; total -= 1
            if alloc_read   > 0 and total > k_main: alloc_read   -= 1; total -= 1
            if alloc_video  > 0 and total > k_main: alloc_video  -= 1; total -= 1

    # INITIAL PICKS (define before using!)
    picks_courses = in_courses[:alloc_course]
    picks_books   = in_books[:alloc_read]
    picks_videos  = in_videos[:alloc_video]

    # Per-modality fallback if focus filtered out everything
    if focus_topic and not picks_books:
        picks_books = [b for b in off_books if profile.matches_item(b.topic_vector)][:max(1, alloc_read)] \
                      or off_books[:max(1, alloc_read)]
    if focus_topic and not picks_videos:
        picks_videos = [v for v in off_videos if profile.matches_item(v.topic_vector)][:max(1, alloc_video)] \
                       or off_videos[:max(1, alloc_video)]
    if focus_topic and not picks_courses:
        picks_courses = [c for c in off_courses if profile.matches_item(c.topic_vector)][:max(1, alloc_course)] \
                        or off_courses[:max(1, alloc_course)]

    # Exploration: pull a few off-topic items across modalities
    explore_pool = off_courses[:k_explore] + off_books[:k_explore] + off_videos[:k_explore]
    random.shuffle(explore_pool)
    explore_picks = explore_pool[:k_explore]

    # Surprise item: something good but outside their usual topics
    surprise_pool = (off_videos[:25] + off_courses[:25] + off_books[:25]) or (videos + courses + books)
    surprise_item = random.choice(surprise_pool) if surprise_pool else None

    # Global safety net (rare but safe)
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
