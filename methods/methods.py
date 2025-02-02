#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 手法選択用クラス

import methods.single_esn_vp as single_esn_vp
import methods.mr_esn_vp as mr_esn_vp

class Methods():
    @staticmethod
    def get(arg):        
        if arg['method'] == 'single_esn_vp':
            return single_esn_vp.Single_ESN_VP(arg)
        elif arg['method'] == 'mr_esn_vp':
            return mr_esn_vp.MR_ESN_VP(arg)
        else:
            return None
