#!/usr/bin/python
from configparser import ConfigParser
import os

def config(filename='database.ini', section='coredata'):
    # Get the absolute path of the file relative to the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, filename)

    # Create a parser
    parser = ConfigParser()

    # Read the config file
    read_files = parser.read(abs_path)
    if not read_files:
        raise Exception(f"Failed to read the configuration file: {abs_path}")

    # Get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception(f"Section {section} not found in the {filename} file.")

    return db
