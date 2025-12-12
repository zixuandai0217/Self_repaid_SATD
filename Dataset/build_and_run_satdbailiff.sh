#!/bin/bash

# Execute Maven build
mvn clean package

# Check if Maven succeeded, exit if failed
if [ $? -ne 0 ]; then
    echo "Maven build failed, script exiting..."
    exit 1
fi

# Run Java program
java -jar ./target/satd-analyzer-jar-with-all-dependencies.jar -r ./repos.csv -d ./mySQL.properties