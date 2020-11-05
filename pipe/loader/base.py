"""
Base class for Pipeline API data loaders.
"""
from abc import ABC, abstractmethod
import copy
from pipe.loader import EVENT


class PipelineLoader(ABC):
    """Interface for PipelineLoaders.
    """
    def _resolve_domains(self, domains, event_type=False):
        pipeline_domain = self._preprocess_domains(domains)
        if event_type and self._validate_event(pipeline_domain):
            return pipeline_domain
        return pipeline_domain

    @staticmethod
    def _preprocess_domains(domains):
        """
        Domain has _fields and specify trading_calendar for computing term
        Verify domains and attempt to compose domains to a composite domain
        where intergrated fields and date_tuples.

        The default implementation is a no-op.
        """
        pipeline_domain = copy.deepcopy(domains[0])
        # print('copy c_test', domains[0], pipeline_domain)
        for domain in domains[1:]:
            pipeline_domain | domain
        # print('before domains', domains[0].domain_window)
        # print('pipeline_domain', pipeline_domain.domain_window)
        return pipeline_domain

    @classmethod
    def _validate_event(cls, domain):
        """
        Verify that the columns of ``events`` can be used by an EventsLoader to
        serve the BoundColumns described by ``next_value_columns`` and
        ``previous_value_columns``.
        """
        event_type = domain.domain_field
        missing = set(event_type) - set(EVENT)
        if missing:
            raise ValueError(
                "EventsLoader missing required columns {missing}.\n"
                "Got Columns: {received}\n"
                "Expected Columns: {required}".format(
                    missing=sorted(missing),
                    received=set(event_type),
                    required=set(EVENT),
                )
            )
        return True

    @abstractmethod
    def load_pipeline_arrays(self, date, sids, data_frequency):
        """
        Load data for ``columns`` as AdjustedArrays.

        Parameters
        ----------
        date : pd.Timestamp or str
            Dates for which data is being requested.
        sids : pd.Int64Index
            Asset identifiers for which data is being requested.
        data_frequency : daily or minute

        Returns
        -------
        arrays : dict[BoundColumn -> zipline.lib.adjusted_array.AdjustedArray]
            Map from column to an AdjustedArray representing a point-in-time
            rolling view over the requested dates for the requested sids.
        """
        raise NotImplementedError
