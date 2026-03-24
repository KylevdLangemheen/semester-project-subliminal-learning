try:
    from beartype.claw import beartype_this_package

    beartype_this_package()
except ImportError:
    pass  # beartype not installed, skip runtime type-checking

__version__ = "0.1.0"
