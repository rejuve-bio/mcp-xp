import logging

def configure_logging():

    root_logger = logging.getLogger()

    # Clear any existing handlers to avoid duplicate or conflicting logs
    root_logger.handlers.clear()

    # Stream handler (CLI)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - [ %(name)s ] - %(levelname)s: %(message)s'))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(stream_handler)

    # Set noisy logs to critical to decrease confusion.
    for noisy_logger in ["bioblend", "requests", "urllib3", "httpx", "neo4j.notifications"]:
        logging.getLogger(noisy_logger).setLevel(logging.CRITICAL)