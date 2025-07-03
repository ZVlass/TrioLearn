
from rest_framework import serializers

class UserFeaturesSerializer(serializers.Serializer):
    id_student               = serializers.IntegerField()
    click_forum              = serializers.IntegerField()
    click_resource           = serializers.IntegerField()
    click_url                = serializers.IntegerField()
    # … add one line per feature column …
    n_active_days            = serializers.IntegerField()
    n_sessions               = serializers.IntegerField()
    total_clicks             = serializers.IntegerField()
    mean_clicks_per_session  = serializers.FloatField()
    avg_score_ratio          = serializers.FloatField()
    pct_on_time              = serializers.FloatField()
    n_submissions            = serializers.IntegerField()


