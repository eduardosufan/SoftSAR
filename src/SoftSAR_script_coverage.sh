#! /bin/bash
coverage erase
coverage run -a Test_ConfigurationManager.py
coverage run -a Test_AirplaneTrajectory.py
coverage run -a Test_Utils.py
coverage run -a Test_SoftSAR.py
coverage report -m --omit=/usr/*
