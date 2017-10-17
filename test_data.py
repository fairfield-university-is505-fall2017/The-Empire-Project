import pandas as pd
from Data import data
import pytest


@pytest.fixture
def thing():
    return 1

def test_data(thing):
    test=data
    assert all(test.columns ==['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','Age','Neighbourhood','Scholarship','Hypertension','Diabetes','Alcoholism','Disabled','SMS Received','No-show'])
    

