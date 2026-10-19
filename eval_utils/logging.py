import logging

# Example placeholders for your custom modules/functions
# from your_module import load_hyperparameters, Custom_Model_Lightning, ToDevice, start_training_lightning
# import your_data_loader as dl

def setup_logger(log_path: str) -> logging.Logger:
    """Configure the logging system to write to both console and file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent duplicate logs if root logger is set

    # If no handlers are attached, add them (avoiding multiple handlers)
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_path, mode='a')  
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(fh)

        # Console handler (optional)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(ch)

    return logger
