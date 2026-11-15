"""The cockpit's own `time` module replaced for one test: a copy of the real one with the attributes given (a
`sleep` that does nothing, or that finishes a checkout), so the test's own waits keep the real module."""
import types


def cockpit_time(cp, test, **attrs):
    fake = types.ModuleType("time")
    fake.__dict__.update(cp.time.__dict__)
    fake.__dict__.update(attrs)
    real = cp.time
    cp.time = fake
    test.addCleanup(setattr, cp, "time", real)
    return fake
