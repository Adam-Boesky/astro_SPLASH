"""Handle the fetching of random forests using pooch."""
import os
import pickle
import hashlib

from pooch import create, Pooch, os_cache

SPLASH_BASEPATH = os.path.dirname(os.path.realpath(__file__))


def calculate_md5(file_path):
    """Calculate the MD5 checksum of a file."""
    print(file_path)
    return hashlib.md5(open(file_path,'rb').read()).hexdigest()


def update_registry():
    """Helper function to update the registry pkl file.
    
    NOTE: This must be ran each time that a random forest is retrained.
    """
    registry = {}
    for fname in [
        f"trained_models/rf_classifier_new_version.pbz2",
        f"trained_models/rf_classifier_old_version.pbz2",
    ]:
        registry[fname] = f'md5:{calculate_md5(os.path.join(SPLASH_BASEPATH, fname))}'
    with open(os.path.join(SPLASH_BASEPATH, 'trained_models/rf_registry.pkl'), 'wb') as f:
        pickle.dump(registry, f)

    return registry


def get_registry(update: bool = False) -> dict:
    """Get the stored registry."""
    if update:
        return update_registry()
    with open(os.path.join(SPLASH_BASEPATH, 'trained_models/rf_registry.pkl'), 'rb') as f:
        return pickle.load(f)


def get_goodboy() -> Pooch:
    """Get the Pooch object used to fetch files."""
    # Define your Pooch object
    return create(
        path=os_cache("SPLASH"),
        base_url="https://raw.githubusercontent.com/Adam-Boesky/astro_SPLASH/main/SPLASH/",
        registry=get_registry(),
    )
