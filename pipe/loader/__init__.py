# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 15:37:47 2019

@author: python
"""
from gateway.driver.fundamental_reader import (
                        MassiveSessionReader,
                        ReleaseSessionReader,
                        HolderSessionReader,
                        StructureSessionReader,
                        GrossSessionReader,
                        MarginSessionReader
                                            )
EVENT = {
        'massive': MassiveSessionReader(),
        'release': ReleaseSessionReader(),
        'holder': HolderSessionReader(),
        'structure': StructureSessionReader(),
        'gross': GrossSessionReader(),
        'margin': MarginSessionReader()
        }
