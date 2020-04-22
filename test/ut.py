# -*- coding:utf-8 -*-

import unittest

def get_formatted_name(first,second):
    name = ('').join([first.capitalize(),second.capitalize()])
    return name

class NamesTestCase(unittest.TestCase):
    """
        所有以test_开头的方法都会自动运行
        assertEqual,assertNotEqual,assertTrue,assertFalse,assertIn,assertNotIn
    """
    def test_first_last_name(self):
        formatted_name = get_formatted_name('janis', 'joplin')
        self.assertEqual(formatted_name,'Janis Joplin')
        unittest.main()