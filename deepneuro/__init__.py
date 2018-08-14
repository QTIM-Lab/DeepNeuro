import warnings

warnings.filterwarnings("ignore", message=".*dtype size changed.*")
warnings.filterwarnings("ignore", message=".*keepdims.*")
warnings.filterwarnings("ignore", message=".*pydicom.*")