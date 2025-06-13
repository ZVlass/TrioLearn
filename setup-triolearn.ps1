# ------------------------------------------
# PowerShell script to scaffold TrioLearn project (simplified)
# ------------------------------------------
# 1. Set your base project directory here:
$base = "C:\Users\jvlas\source\repos\TrioLearn"

# 2. List of subfolders relative to $base
$folders = @(
    ".vscode",
    "docker",
    "data",
    "data\raw",
    "data\external",
    "data\interim",
    "data\processed",
    "notebooks",
    "src",
    "src\data",
    "src\features",
    "src\models",
    "src\evaluation",
    "src\utils",
    "src\services",
    "scripts",
    "outputs",
    "outputs\models",
    "outputs\figures",
    "outputs\logs",
    "outputs\reports"
)

# 3. List of files to create (relative to $base). These will be empty files.
$files = @(
    ".gitignore",
    "README.md",
    "environment.yml",
    "requirements.txt",
    ".env",
    "docker\Dockerfile",
    "docker\docker-compose.yml",
    "notebooks\01_data_exploration.ipynb",
    "notebooks\02_text_preprocessing.ipynb",
    "notebooks\03_embedding_training.ipynb",
    "notebooks\04_topic_modeling_sentiment.ipynb",
    "notebooks\05_user_modeling_oulad.ipynb",
    "notebooks\06_prototype_recommendation.ipynb",
    "notebooks\07_evaluation_metrics.ipynb",
    "src\__init__.py",
    "src\data\load_data.py",
    "src\data\preprocess.py",
    "src\data\cache_utils.py",
    "src\features\embeddings.py",
    "src\features\topic_model.py",
    "src\features\sentiment.py",
    "src\models\knowledge_tracing.py",
    "src\models\graph_builder.py",
    "src\models\gnn.py",
    "src\models\recommender.py",
    "src\evaluation\metrics.py",
    "src\evaluation\significance.py",
    "src\evaluation\evaluation_pipeline.py",
    "src\utils\config.py",
    "src\utils\logging.py",
    "src\utils\viz.py",
    "src\services\youtube_api.py",
    "src\services\google_books_api.py",
    "src\services\coursera_api.py",
    "scripts\preprocess_all.py",
    "scripts\build_graph.py",
    "scripts\train_gnn.py",
    "scripts\recommend.py",
    "scripts\evaluate.py",
    ".vscode\settings.json",
    ".vscode\launch.json"
)

# 4. Create base folder if it doesn't exist
if (-Not (Test-Path -Path $base)) {
    New-Item -ItemType Directory -Path $base -Force | Out-Null
    Write-Host "Created base folder: $base"
} else {
    Write-Host "Base folder already exists: $base"
}

# 5. Create all subfolders
foreach ($folder in $folders) {
    $fullPath = Join-Path $base $folder
    if (-Not (Test-Path -Path $fullPath)) {
        New-Item -ItemType Directory -Path $fullPath -Force | Out-Null
        Write-Host "Created folder: $fullPath"
    } else {
        Write-Host "Folder already exists: $fullPath"
    }
}

# 6. Create empty files
foreach ($relativePath in $files) {
    $filePath = Join-Path $base $relativePath
    $dirPath = Split-Path $filePath -Parent

    # Ensure the directory exists
    if (-Not (Test-Path -Path $dirPath)) {
        New-Item -ItemType Directory -Path $dirPath -Force | Out-Null
        Write-Host "Created directory for file: $dirPath"
    }

    if (-Not (Test-Path -Path $filePath)) {
        New-Item -ItemType File -Path $filePath -Force | Out-Null
        Write-Host "Created empty file: $filePath"
    } else {
        Write-Host "File already exists: $filePath"
    }
}

Write-Host "Scaffold complete at $base"
