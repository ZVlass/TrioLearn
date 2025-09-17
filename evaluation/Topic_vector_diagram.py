
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 140

def add_box(ax, xy, w, h, title, fields, fc="#f5f5f5"):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor="#222222", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0]+0.02, xy[1]+h-0.18, title, ha='left', va='top', fontsize=10, fontweight='bold')
    y = xy[1]+h-0.38
    for f in fields:
        ax.text(xy[0]+0.04, y, f, ha='left', va='top', fontsize=8)
        y -= 0.22
    return rect

def arrow(ax, xy_from, xy_to, text=None):
    arr = FancyArrow(xy_from[0], xy_from[1], xy_to[0]-xy_from[0], xy_to[1]-xy_from[1],
                     width=0.002, length_includes_head=True, head_width=0.06, head_length=0.09,
                     color="#444444")
    ax.add_patch(arr)
    if text:
        mx, my = (xy_from[0]+xy_to[0])/2, (xy_from[1]+xy_to[1])/2
        ax.text(mx, my+0.05, text, ha='center', va='bottom', fontsize=8)

fig, ax = plt.subplots(figsize=(10,5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')

b = add_box(ax, (0.6, 4.2), 3.2, 2.2, "book_topic_vectors.parquet", [
    "id",
    "topic_00 … topic_K",
    "(merged fields added later:)",
    "title, external_id, modality='book'"
], fc="#e8f0fe")

c = add_box(ax, (0.6, 2.2), 3.2, 2.2, "course_topic_vectors.parquet", [
    "id",
    "topic_00 … topic_K",
    "(merged fields added later:)",
    "title, external_id, modality='course'"
], fc="#e6f4ea")

v = add_box(ax, (0.6, 0.2), 3.2, 2.2, "video_topic_vectors.parquet", [
    "id",
    "topic_00 … topic_K",
    "(merged fields added later:)",
    "title, external_id, modality='video'"
], fc="#fff4e5")

m = add_box(ax, (6.0, 1.6), 4.4, 3.6, "all_topic_vectors.parquet", [
    "modality  (book|course|video)",
    "external_id",
    "title",
    "topic_vector  (list[float], len=K)",
    "num_topics"
], fc="#fde7e9")

arrow(ax, (3.9, 5.0), (6.0, 4.5), "merge_topics.py")
arrow(ax, (3.9, 2.7), (6.0, 3.4), "merge_topics.py")
arrow(ax, (3.9, 0.1), (6.0, 2.3), "merge_topics.py")

ax.text(0.6, -1.4, "TrioLearn ERD – Topic Vector Files (per modality → unified schema)\n"
                   "Fields in italics appear after merge (normalize id→external_id, add modality/title, pack topic_vector).",
        ha='left', va='top', fontsize=8)

plt.tight_layout()
out_path = "evaluation/images/TrioLearn_ERD_TopicVectors.png"
plt.savefig(out_path, bbox_inches="tight")
out_path
