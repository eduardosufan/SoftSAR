'''
Created on 23 ene. 2017

@author: esufan
@summary: Module containing ConfigurationManager class for read configuration of SoftSAR.

-Attributes: -.
-Methods: -.
-Classes: ConfigurationManager

'''

import xml.etree.ElementTree as ET


class ConfigurationManager(object):
    """
    Read configuration data from xml file.
    Note: elements in xml file must be unique.

    Attributes
    ----------
    None.

    Methods
    -------
    get_string_parameter
    get_int_parameter
    get_float_parameter
    get_object_parameter
    parameter_exists
    """

    def __init__(self, input_file):
        """
        Constructor
        """
        self.tree = ET.parse(input_file)
        self.root = self.tree.getroot()
    
    def get_string_parameter(self, parameter):
        """
        Read input data from xml file as String.
        
        Details
        -------
        Element in xml file must be unique.
        
        Parameters
        ----------
        parameter: string
         String containing name of parameter in xml file.
        
        Returns
        -------
        value: string
         String value of the Element.
        """
        return self.get_object_parameter(parameter).text
    
    def get_int_parameter(self, parameter):
        """
        Read input data from xml file as int.
        
        Details
        -------
        Element in xml file must be unique.
        
        Parameters
        ----------
        parameter: string
         String containing name of parameter in xml file.
        
        Returns
        -------
        value: int
         Int value of the Element.
        """
        return int(self.get_object_parameter(parameter).text)
    
    def get_float_parameter(self, parameter):
        """
        Read input data from xml file as float.
        
        Details
        -------
        Element in xml file must be unique.
        
        Parameters
        ----------
        parameter: string
         String containing name of parameter in xml file.
        
        Returns
        -------
        value: float
         Float value of the Element.
        """
        return float(self.get_object_parameter(parameter).text)
        
    def get_object_parameter(self, parameter):
        """
        Read input data from xml file as object.
        
        Details
        -------
        Element in xml file must be unique.
        
        Parameters
        ----------
        parameter: string
         String containing name of parameter in xml file.
        
        Returns
        -------
        value: object
         The object ElementTree Element.
        """
        try:
            return self.root.findall(parameter)[0]
        except IndexError:
            raise Exception("Element not found in xml")


#****************************************************************************************************
#****************************************************************************************************

    def parameter_exists(self, parameter):
        """
        Verify if element exists into xml file.
        
        Details
        -------
        Element in xml file must be unique.
        
        Parameters
        ----------
        parameter: string
         String containing name of element in xml file.
        
        Returns
        -------
        True if element exists, false if not.
        """
        try:
            self.root.findall(parameter)[0]
            return True
        except IndexError:
            return False 

