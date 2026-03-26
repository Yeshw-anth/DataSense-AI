import pytest
from unittest.mock import patch
from agents.planner_agent import PlannerAgent
import pandas as pd
import os
import json
from pipelines.main_pipeline import graph_app

# Define a sample CSV for testing
SAMPLE_CSV_DATA = "product,category,sales\nLaptop,Electronics,1500\nMouse,Electronics,25\nKeyboard,Electronics,75\nDesk,Furniture,300\nChair,Furniture,150"