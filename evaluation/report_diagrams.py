import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 140

def add_box(ax, xy, w, h, text, fc="#f5f5f5", ec="#222222"):
    rect = Rectangle(xy, w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center', fontsize=9, wrap=True)
    return rect

def add_arrow(ax, xy_from, xy_to, text=None):
    arrow = FancyArrow(xy_from[0], xy_from[1], xy_to[0]-xy_from[0], xy_to[1]-xy_from[1],
                       width=0.002, length_includes_head=True, head_width=0.05, head_length=0.08,
                       color="#444444")
    ax.add_patch(arrow)
    if text:
        mx, my = (xy_from[0]+xy_to[0])/2, (xy_from[1]+xy_to[1])/2
        ax.text(mx, my+0.03, text, ha='center', va='bottom', fontsize=8)
    return arrow

# 1) Architecture Diagram
fig, ax = plt.subplots(figsize=(9,5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
add_box(ax, (0.5, 4.6), 9, 1.0, "Application Layer\n(Django REST API, Web UI)", fc="#e8f0fe")
add_box(ax, (0.5, 3.1), 9, 1.0, "Recommendation Engine\n(Embedding retrieval, Topic rerank, Tri‑modal bundling, Surprise)", fc="#e6f4ea")
add_box(ax, (0.5, 1.6), 9, 1.0, "Data Processing\n(Cleaning, Normalization, SBERT, LDA Topics)", fc="#fff4e5")
add_box(ax, (0.5, 0.1), 9, 1.0, "Data Ingestion & Storage\n(Coursera/edX, Google Books, YouTube, OULAD → CSV/Parquet, PostgreSQL)", fc="#fde7e9")
add_box(ax, (1.0, 3.25), 2.2, 0.7, "Vector Store\n(Embeddings .npy/\nFAISS index)")
add_box(ax, (3.4, 3.25), 2.2, 0.7, "Topic Store\n(LDA θ per item,\nTop words)")
add_box(ax, (5.8, 3.25), 2.2, 0.7, "Rules & Signals\n(Difficulty, Popularity,\nFilters)")
add_box(ax, (8.2, 3.25), 1.0, 0.7, "Bundles")
add_arrow(ax, (5.0, 5.1), (5.0, 4.1), "API calls")
add_arrow(ax, (5.0, 4.1), (5.0, 2.6), "Ranking")
add_arrow(ax, (5.0, 2.6), (5.0, 1.1), "Features")
add_arrow(ax, (5.0, 1.1), (5.0, 0.6), "ETL")
plt.tight_layout()
arch_path = "/reports/images/TrioLearn_Architecture.png"
plt.savefig(arch_path)
plt.close(fig)

# 2) Data Pipeline Diagram
fig, ax = plt.subplots(figsize=(10,5.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
add_box(ax, (0.4, 4.8), 2.8, 1.0, "Raw Data\nCoursera/edX\nBooks (Google/OpenLib)\nYouTube\nOULAD", fc="#fde7e9")
add_box(ax, (3.6, 4.8), 2.8, 1.0, "Cleaning/Normalization\nLowercase, URLs, NaNs,\nDifficulty mapping", fc="#fff4e5")
add_box(ax, (6.8, 4.8), 2.8, 1.0, "Text Fields\ntext_for_embedding\n+ topic corpus", fc="#fff4e5")
add_box(ax, (10.0, 4.8), 2.8, 1.0, "Persist\nCSV/Parquet\nInterim", fc="#fff4e5")
add_arrow(ax, (3.2, 5.3), (3.6, 5.3))
add_arrow(ax, (6.4, 5.3), (6.8, 5.3))
add_arrow(ax, (9.6, 5.3), (10.0, 5.3))
add_box(ax, (1.2, 2.5), 3.2, 1.0, "SBERT Embeddings\n(all‑MiniLM‑L6‑v2)\nNormalize → .npy", fc="#e6f4ea")
add_box(ax, (5.2, 2.5), 3.2, 1.0, "LDA Topics\nCountVectorizer + LDA\nθ vectors + top words", fc="#e6f4ea")
add_box(ax, (9.2, 2.5), 3.2, 1.0, "Merged Topics\nall_topic_vectors.parquet", fc="#e6f4ea")
add_arrow(ax, (2.0, 4.8), (2.8, 3.5), "to SBERT")
add_arrow(ax, (7.0, 4.8), (6.8, 3.5), "to LDA")
add_arrow(ax, (8.4, 3.0), (9.2, 3.0))
plt.tight_layout()
pipeline_path = "/report/images/TrioLearn_DataPipeline.png"
plt.savefig(pipeline_path)
plt.close(fig)

# 3) Tri‑Modal Flow Diagram
fig, ax = plt.subplots(figsize=(10,5))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off')
add_box(ax, (0.5, 2.2), 2.2, 1.0, "User Query\n(text/profile)", fc="#e8f0fe")
add_box(ax, (3.0, 2.2), 2.2, 1.0, "Embed Query\nSBERT", fc="#e6f4ea")
add_box(ax, (5.5, 4.0), 2.2, 0.8, "Course\nEmbeddings")
add_box(ax, (5.5, 2.2), 2.2, 0.8, "Book\nEmbeddings")
add_box(ax, (5.5, 0.4), 2.2, 0.8, "Video\nEmbeddings")
add_box(ax, (8.2, 2.2), 2.2, 1.0, "Ranking\ncosine + topics\n+ difficulty", fc="#e6f4ea")
add_box(ax, (10.8, 2.2), 2.2, 1.0, "Bundles\nCourse + Book + Video\n(+ Surprise)", fc="#e6f4ea")
add_arrow(ax, (2.7, 2.7), (3.0, 2.7))
add_arrow(ax, (5.2, 2.7), (5.5, 3.8), "to courses")
add_arrow(ax, (5.2, 2.7), (5.5, 2.6), "to books")
add_arrow(ax, (5.2, 2.7), (5.5, 1.0), "to videos")
add_arrow(ax, (7.7, 2.7), (8.2, 2.7))
add_arrow(ax, (10.4, 2.7), (10.8, 2.7))
add_box(ax, (8.2, 0.4), 2.2, 0.8, "Surprise Selector\nrelevance × novelty")
add_arrow(ax, (9.3, 1.2), (11.0, 2.0))
plt.tight_layout()
trimodal_path = "/reports/images/TrioLearn_TriModalFlow.png"
plt.savefig(trimodal_path)
plt.close(fig)

# 4) Evaluation Flow Diagram
fig, ax = plt.subplots(figsize=(9.5,5))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
add_box(ax, (0.5, 4.6), 2.6, 1.0, "Historical Logs\n(OULAD / Interactions)", fc="#fde7e9")
add_box(ax, (0.5, 3.1), 2.6, 1.0, "Train/Test Split\n(time-based holdout)", fc="#fff4e5")
add_box(ax, (0.5, 1.6), 2.6, 1.0, "Model Variants\n(SBERT, +Topics, Hybrid)", fc="#e6f4ea")
add_arrow(ax, (1.8, 5.1), (1.8, 4.1))
add_arrow(ax, (1.8, 3.6), (1.8, 2.6))
add_box(ax, (3.8, 4.6), 2.8, 1.0, "Ranking Metrics\nPrecision@k, Recall@k,\nAUC", fc="#e8f0fe")
add_box(ax, (3.8, 3.1), 2.8, 1.0, "Intrinsic Metrics\nNovelty, Diversity", fc="#e8f0fe")
add_box(ax, (3.8, 1.6), 2.8, 1.0, "Significance Tests\nA/B plan, CI", fc="#e8f0fe")
add_arrow(ax, (3.3, 5.1), (3.8, 5.1))
add_arrow(ax, (3.3, 3.6), (3.8, 3.6))
add_arrow(ax, (3.3, 2.1), (3.8, 2.1))
add_box(ax, (7.2, 3.1), 2.6, 1.0, "Reports\nCSV tables + PNG plots\n(for Appendix)", fc="#e6f4ea")
add_arrow(ax, (6.6, 5.1), (7.2, 3.6))
add_arrow(ax, (6.6, 3.6), (7.2, 3.6))
add_arrow(ax, (6.6, 2.1), (7.2, 2.6))
plt.tight_layout()
eval_path = "/report/images/TrioLearn_EvaluationFlow.png"
plt.savefig(eval_path)
plt.close(fig)

(arch_path, pipeline_path, trimodal_path, eval_path)
