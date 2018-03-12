#! /usr/bin/env python
import unittest

tests = unittest.TestLoader().discover('.')
unittest.runner.TextTestRunner().run(tests)
