
import networkx as nx,pandas as pd
from weakref import WeakValueDictionary
from interface import default, implements, Interface

import glob, uuid

class IDomain(Interface):
    """
    A domain defines two things:

    1. A calendar defining the dates to which the pipeline's inputs and outputs
       should be aligned. The calendar is represented concretely by a pandas
       DatetimeIndex.

    2. The set of assets that the pipeline should compute over. Right now, the only
       supported way of representing this set is with a two-character country code
       describing the country of assets over which the pipeline should compute. In
       the future, we expect to expand this functionality to include more general
       concepts.
    """
    def all_sessions(self,sdate,edate):
        """
        Get all trading sessions for the calendar of this domain.

        This determines the row labels of Pipeline outputs for pipelines run on
        this domain.

        Returns
        -------
        sessions : pd.DatetimeIndex
            An array of all session labels for this domain.
        """

    def all_assets(self,category = 'stock'):
        """
            Get all assets
        """

    def data_query_cutoff_for_sessions(self, sessions):
        raise NotImplementedError(
            "Can't compute data query cutoff times for generic domain.",
        )

class Domain(implements(IDomain)):
    """
    This module defines the interface and implementations of Pipeline domains.

    A domain represents a set of labels for the arrays computed by a Pipeline.
    Currently, this means that a domain defines two things:

    1. A calendar defining the dates to which the pipeline's inputs and outputs
       should be aligned. The calendar is represented concretely by a pandas
       DatetimeIndex.

    2. The set of assets that the pipeline should compute over. Right now, the only
       supported way of representing this set is with a two-character country code
       describing the country of assets over which the pipeline should compute. In
       the future, we expect to expand this functionality to include more general
       concepts.
    """
    __doc__ = """ """
    __name__ = "Domain"
    __qualname__ = "zipline.pipeline.domain.Domain"

    def all_session(self,s,e):
        raise NotImplementedError

    def all_assets(self,category= 'stock'):
        raise NotImplementedError

    def roll_forward(self, dt):
        """
        Given a date, align it to the calendar of the pipeline's domain.

        Parameters
        ----------
        dt : pd.Timestamp

        Returns
        -------
        pd.Timestamp
        """
        dt = pd.Timestamp(dt, tz='UTC')

        trading_days = self.all_sessions()
        try:
            return trading_days[trading_days.searchsorted(dt)]
        except IndexError:
            raise ValueError(
                "Date {} was past the last session for domain {}. "
                "The last session for this domain is {}.".format(
                    dt.date(),
                    self,
                    trading_days[-1].date()
                )
            )
