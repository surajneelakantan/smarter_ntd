def is_streamlit() -> bool:
    """
    This will be executed in the CLI mode. For the Streamlit mode, see `backend_bootstrap.py`.
    """
    return False


def prepare_buttons(labels: list[str]) -> None:
    """
    This will be executed in the CLI mode. For the Streamlit mode, see `backend_bootstrap.py`.
    """
    pass


def show_download_button(label: str, url: str) -> None:
    """
    This will be executed in the CLI mode. For the Streamlit mode, see `backend_bootstrap.py`.
    """
    print(f'{label}: {url}')