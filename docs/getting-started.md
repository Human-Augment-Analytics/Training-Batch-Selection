# Getting Started

## Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd Training-Batch-Selection

# Set up Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Development Workflow
1. **Create a branch** for your work:
   ```bash
   git checkout -b <Author's GaTech Account>/your-feature-name
   ```

2. **Format your code** before committing:
   ```bash
   black .
   ```

3. **Run basic checks**:
   ```bash
   # Add any linting/testing commands here
   python -m pytest  # if tests exist
   ```

4. **Commit and push**:
   ```bash
   git add .
   git commit -m "Add your feature description"
   git push origin feature/your-feature-name
   ```

## Project Structure
- `trainer/data/` - Data loading and preprocessing
- `trainer/model/` - Model architecture
- `trainer/pipelines/` - Training and evaluation pipelines
- `docs/` - Documentation and standards
