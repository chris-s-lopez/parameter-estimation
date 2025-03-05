#!/bin/bash

echo "Running unit tests..."
python3 -m unittest discover -s tests -p "test_*.py"

if [ $? -eq 0 ]; then
    echo "✅ All tests passed successfully!"
else
    echo "❌ Some tests failed. Check the output above."
fi
