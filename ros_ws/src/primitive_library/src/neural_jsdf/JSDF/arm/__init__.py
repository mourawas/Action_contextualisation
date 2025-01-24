from .wrapper import iiwa_JSDF
from neural_jsdf.models import *
from neural_jsdf import models
import sys
sys.modules['models'] = models
