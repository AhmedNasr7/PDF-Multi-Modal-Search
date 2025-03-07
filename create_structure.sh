#!/bin/bash

# Define project root
PROJECT_ROOT="."

# Create the main project directory
mkdir -p $PROJECT_ROOT

# Create module directories
mkdir -p $PROJECT_ROOT/modules
mkdir -p $PROJECT_ROOT/server

# Create empty Python module files
touch $PROJECT_ROOT/modules/__init__.py
touch $PROJECT_ROOT/modules/document_parser.py
touch $PROJECT_ROOT/modules/text_processor.py
touch $PROJECT_ROOT/modules/vector_db.py
touch $PROJECT_ROOT/modules/reranker.py
touch $PROJECT_ROOT/modules/vlm_service.py
touch $PROJECT_ROOT/modules/query_pipeline.py

# Create server-related files
touch $PROJECT_ROOT/server/vlm_server.py

# Create main execution files
touch $PROJECT_ROOT/main.py
touch $PROJECT_ROOT/app.py

# Create project metadata files
touch $PROJECT_ROOT/requirements.txt
touch $PROJECT_ROOT/README.md

# Output project structure
echo "✅ Project structure created successfully!"
tree $PROJECT_ROOT
