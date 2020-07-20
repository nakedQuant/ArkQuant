# -*- coding:utf-8 -*-

import unittest

class NamesTestCase(unittest.TestCase):
    """
        所有以test_开头的方法都会自动运行
        assertEqual,assertNotEqual,assertTrue,assertFalse,assertIn,assertNotIn
        setUp -- called before test method ; setUpClass --A  class method called before tests in an individual class are run
    """
    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_first_last_name(self):
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass