"""Marco observe/act del libro de 50 dólares."""

__all__ = ["AgenteInversor", "Book", "PoliticaInversor"]


def __getattr__(name):
    """Carga los componentes solo cuando se necesitan."""
    if name == "AgenteInversor":
        from src.agent.investor import AgenteInversor
        return AgenteInversor

    if name == "Book":
        from src.agent.book import Book
        return Book

    if name == "PoliticaInversor":
        from src.agent.policy import PoliticaInversor
        return PoliticaInversor

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
