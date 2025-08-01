def extract_features(query, learner_level="intermediate"):
    """
    Extracts simple features for ML model input.

    Args:
        query (str): Natural language user query
        learner_level (str): Learning stage

    Returns:
        dict: Feature name → value


    Temporary fix to match features the trained model expects:
    ['course_count', 'reading_count', 'video_count']

    """
    return {
        "course_count": 10,
        "reading_count": 5,
        "video_count": 20
    }
