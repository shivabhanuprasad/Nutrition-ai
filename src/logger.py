#import python's built-in logging library
import logging
#import os utilities for filesystem operations (like making folder)
import os
#importing datatime to timestamp lof files with today's data
from datetime import datetime
#name of the folder where logs will be stored
LOGS_DIR= "logs"
#create the logs folder if it dosen't already exist
os.makedirs(LOGS_DIR,exist_ok=True)
#build a log file path like : logs/log_2025-08-14.log (changes daily)
LOG_FILE=os.path.join(
    LOGS_DIR,
    f"log_{datetime.now().strftime('%Y-%m-%d')}.log"
)
#config the ROOT logger once for the whole program
logging.basicConfig(
    #write all the logs for the file
    filename=LOG_FILE,
    #log message format
    # - %(asctime)s : timestamp
    # - %(levelname)s : logs level (INFO,ERROR,etc.)
    # - %(message)s : the log message text 
    format='%(asctime)s-%(levelname)s-%(message)s',
    #minimum level to read record
    level=logging.INFO

)
def get_logger(name):
    """
    Returns a named logger that inherits the root configuration above.
    Use different names per module (e.g., __name__) to identify sources.
    """
    # Get (or create) a logger with the given name
    logger = logging.getLogger(name)

    # Ensure this logger emits INFO and above (can be customized per logger)
    logger.setLevel(logging.INFO)

    # Return the configured named logger
    return logger