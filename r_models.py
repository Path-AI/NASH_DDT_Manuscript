import pandas as pd
import numpy as np
import json

import rpy2.robjects as ro
import rpy2.robjects.packages as rpack
from rpy2.robjects import pandas2ri
from rpy2.robjects import numpy2ri
import rpy2.rlike.container as rlc
from rpy2.robjects.packages import importr

class R:

    def __init__(self):
        self.r = ro.r
        self.base = self._library('base')
        self.stats = self._library('stats')
        self.jsonio = self._library('RJSONIO')

    def _library(self, pack_name):
        return rpack.importr(pack_name)

    def _to_dict(self, rlist):
        rjson = self.jsonio.toJSON(rlist)
        return json.loads( rjson[0] )

class Rnumpy(R):
    def __init__(self):
        pass
    def __enter__(self):
        numpy2ri.activate()
    def __exit__(self, *args):
        numpy2ri.deactivate()

class Rpandas(R):
    def __init__(self):
        pass
    def __enter__(self):
        pandas2ri.activate()
    def __exit__(self, *args):
        pandas2ri.deactivate()

class Rplot(R):
    def __init__(self):
        self.plot = self._library('grDevices')
    def __enter__(self):
        return self.plot
    def __exit__(self, *args):
        self.plot.dev_off()

class vanElterenTest(R):
    def __init__(self):
        super().__init__()
        self.lib = self._library('sanon')

    def test(self, df, outcome_col, treat_col, treat_target, treat_ref, strata_col=None):
        # move data from df to numpy

        df = df[[outcome_col,treat_col,strata_col]].copy()
        df = df[df[treat_col].isin([treat_target, treat_ref])]
        df = df.dropna()

        outcome = df[outcome_col].to_numpy()
        treat = df[treat_col].to_numpy()
        strata = df[strata_col].to_numpy()
        ref = treat_ref

        with Rnumpy():
            self.output = self.lib.sanon(outcome=outcome,group=treat,strt=strata,ref=ref)
            self.stats_dict = super()._to_dict(self.output)
            self.p = self.output.rx2('p')[0][0]

        return self.p

# class confintervals(R):
#     def __init__(self):
#         super().__init__()
#         self.lib = self._library('Hmisc')
    
def get_confintervals(x,n,alpha = 0.05, method = "wilson"):
    hmisc = importr("Hmisc")
    with Rnumpy():
        conf_intervals = list(hmisc.binconf(x,n,alpha, method))
        return [conf_intervals[1],conf_intervals[2]]

    
if __name__ == "__main__":

    df = pd.DataFrame({
    'x': [0,1+2,2,3+2,4,6+2,7,8+2,9,10+2],
    'g': ['0','1','0','1','0','1','0','1','0','1'],
    's': [0,0,0,0,0,1,1,1,1,1]
    })

    mod = vanElterenTest()
    mod.test(df, 'x', 'g', '0','1', strata_col='s')
    print(mod.p)
    print(mod.out_dict)
