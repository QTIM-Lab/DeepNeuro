# A wrapper function to monitor coverage through all tests

from coverage import coverage
import pytest

cov = coverage(omit='.tox*')
cov.start()

# Tests to run
# Pytest will crawl through the project directory for test files.
pytest.main()

cov.stop()
cov.save()