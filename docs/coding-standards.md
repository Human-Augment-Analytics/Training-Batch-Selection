# Coding Standards

## Python Code Formatting
- Use **Black** for automatic code formatting
- Run `black .` before committing
- Line length: 88 characters (Black default)

## Code Style
- Use type hints for function parameters and return values
- Follow PEP 8 naming conventions:
  - Variables and functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

## Documentation
- Use docstrings for all public functions and classes
- Keep inline comments minimal and focused on "why", not "what"

## Error Handling
- Use specific exception types
- Avoid bare `except:` clauses
- Log errors appropriately

## Testing
- Write tests for new functionality
- Use descriptive test names
- Keep tests isolated and deterministic
