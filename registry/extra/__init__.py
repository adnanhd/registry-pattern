"""Optional, opt-in batteries: concrete meters and reporters.

The core (:mod:`registry.meters`, :mod:`registry.reporters`) ships only the
buses and their base classes. Import the concrete implementations from here::

    from registry.extra.meters import CPUMeter, MemoryMeter
    from registry.extra.reporters import JournalReporter, HTTPDashboardReporter
"""
