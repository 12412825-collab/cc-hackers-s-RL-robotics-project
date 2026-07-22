import logging

logger = logging.getLogger(__name__)

class BookDatabase:
    """
    Mock database mapping book titles to shelf locations.
    In a real SLAM system, these would be (x,y) coordinates.
    For this prototype, it might map to a specific track segment or distance.
    """
    def __init__(self):
        self.db = {
            "The Great Gatsby": "Shelf A1",
            "1984": "Shelf B2",
            "To Kill a Mockingbird": "Shelf C3"
        }
        self.current_target_book = None
        
    def request_book(self, title):
        if title in self.db:
            self.current_target_book = title
            logger.info(f"Book requested: {title}. Target location: {self.db[title]}")
            return self.db[title]
        else:
            logger.warning(f"Book '{title}' not found in database.")
            return None

    def run(self):
        # In a real system, this would output the current target coordinate to the navigation AI
        return self.current_target_book


class NoiseDetector:
    """
    Detects loud noises and triggers an alert.
    Requires a sound sensor (e.g. KY-037) connected to the Arduino or Raspberry Pi.
    """
    def __init__(self, threshold=80.0):
        self.threshold = threshold
        self.alert_active = False

    def run(self, sound_level):
        """
        Takes in a sound_level reading (e.g. from an ADC or Arduino serial).
        Outputs True if an alert should be triggered (LED/speaker).
        """
        if sound_level is None:
            return False
            
        if sound_level > self.threshold:
            if not self.alert_active:
                logger.warning(f"LOUD NOISE DETECTED ({sound_level} > {self.threshold})! Triggering Alert.")
                self.alert_active = True
        else:
            self.alert_active = False
            
        return self.alert_active
